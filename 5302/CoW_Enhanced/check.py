#!/usr/bin/env python3
"""Validate the local CMR/CHAOS-style NIfTI dataset layout."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - depends on local environment
    sitk = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - depends on local environment
    nib = None


@dataclass(frozen=True)
class Layout:
    name: str
    image_dir: Path
    label_dir: Path
    image_prefix: str
    label_prefixes: tuple[str, ...]
    supervoxel_dir: Path | None = None
    supervoxel_prefix: str = "superpix-MIDDLE_"
    expected_labels: frozenset[int] = frozenset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check CMR/CHAOS NIfTI dataset files for pairing, readability, shapes, metadata and labels."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("CMR"),
        help="Dataset root directory. Default: CMR",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "cmr", "chaost2", "split"),
        default="auto",
        help="Dataset layout to check. Default: auto",
    )
    parser.add_argument(
        "--check-supervoxels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also check supervoxel files when the layout defines them. Default: true",
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Treat unexpected label values as errors instead of warnings.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Only read the first N cases after pairing checks. Useful for a quick smoke test.",
    )
    return parser.parse_args()


def collect_case_files(directory: Path, prefixes: Iterable[str]) -> tuple[dict[int, Path], list[str]]:
    files: dict[int, Path] = {}
    problems: list[str] = []
    prefixes = tuple(prefixes)

    if not directory.exists():
        return files, [f"Missing directory: {directory}"]

    for path in sorted(directory.glob("*.nii.gz")):
        case_id = parse_case_id(path.name, prefixes)
        if case_id is None:
            continue
        if case_id in files:
            problems.append(f"Duplicate case id {case_id} in {directory}: {files[case_id].name}, {path.name}")
            continue
        files[case_id] = path

    return files, problems


def parse_case_id(filename: str, prefixes: Iterable[str]) -> int | None:
    for prefix in prefixes:
        if filename.startswith(prefix) and filename.endswith(".nii.gz"):
            raw_id = filename[len(prefix) : -len(".nii.gz")]
            return int(raw_id) if raw_id.isdigit() else None
    return None


def build_layouts(root: Path) -> dict[str, list[Layout]]:
    return {
        "cmr": [
            Layout(
                name="cmr_MR_normalized",
                image_dir=root / "cmr_MR_normalized",
                label_dir=root / "cmr_MR_normalized",
                image_prefix="image_",
                label_prefixes=("label_",),
                supervoxel_dir=root / "supervoxels_1000",
                expected_labels=frozenset({0, 1, 2, 3}),
            )
        ],
        "chaost2": [
            Layout(
                name="chaos_MR_T2_normalized",
                image_dir=root / "chaos_MR_T2_normalized",
                label_dir=root / "chaos_MR_T2_normalized",
                image_prefix="image_",
                label_prefixes=("label_",),
                supervoxel_dir=root / "supervoxels_5000",
                expected_labels=frozenset({0, 1, 2, 3, 4, 200, 500, 600}),
            )
        ],
        "split": [
            Layout(
                name="split train",
                image_dir=root / "images",
                label_dir=root / "labels",
                image_prefix="",
                label_prefixes=("label_", ""),
                expected_labels=frozenset({0, 1, 2, 3, 4, 200, 500, 600}),
            ),
            Layout(
                name="split test",
                image_dir=root / "images_test",
                label_dir=root / "labels_test",
                image_prefix="",
                label_prefixes=("label_", ""),
                expected_labels=frozenset({0, 1, 2, 3, 4, 200, 500, 600}),
            ),
        ],
    }


def choose_layouts(root: Path, requested: str) -> list[Layout]:
    layouts = build_layouts(root)
    if requested != "auto":
        return layouts[requested]

    detected: list[Layout] = []
    for candidates in layouts.values():
        for layout in candidates:
            if layout.image_dir.exists() or layout.label_dir.exists():
                detected.append(layout)
    return detected


def read_image(path: Path):
    if sitk is not None:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        return image, array

    if nib is not None:
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
        return image, array

    raise RuntimeError("Install SimpleITK or nibabel to read .nii.gz files.")


def same_metadata(left, right) -> bool:
    if sitk is not None and hasattr(left, "GetSpacing") and hasattr(right, "GetSpacing"):
        return (
            left.GetSpacing() == right.GetSpacing()
            and left.GetOrigin() == right.GetOrigin()
            and left.GetDirection() == right.GetDirection()
        )

    return np.allclose(left.affine, right.affine)


def check_layout(layout: Layout, *, check_supervoxels: bool, strict_labels: bool, max_cases: int | None) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    label_value_counts: Counter[int] = Counter()

    images, image_problems = collect_case_files(layout.image_dir, (layout.image_prefix,))
    labels, label_problems = collect_case_files(layout.label_dir, layout.label_prefixes)
    errors.extend(image_problems)
    errors.extend(label_problems)

    supervoxels: dict[int, Path] = {}
    if check_supervoxels and layout.supervoxel_dir is not None:
        supervoxels, supervoxel_problems = collect_case_files(layout.supervoxel_dir, (layout.supervoxel_prefix,))
        errors.extend(supervoxel_problems)

    image_ids = set(images)
    label_ids = set(labels)
    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)
    if missing_labels:
        errors.append(f"Missing labels for case ids: { compact_ids(missing_labels) }")
    if missing_images:
        errors.append(f"Missing images for case ids: { compact_ids(missing_images) }")

    if check_supervoxels and layout.supervoxel_dir is not None:
        supervoxel_ids = set(supervoxels)
        missing_supervoxels = sorted(image_ids - supervoxel_ids)
        extra_supervoxels = sorted(supervoxel_ids - image_ids)
        if missing_supervoxels:
            errors.append(f"Missing supervoxels for case ids: { compact_ids(missing_supervoxels) }")
        if extra_supervoxels:
            warnings.append(f"Extra supervoxels without images: { compact_ids(extra_supervoxels) }")

    paired_ids = sorted(image_ids & label_ids)
    if check_supervoxels and layout.supervoxel_dir is not None:
        paired_ids = sorted(set(paired_ids) & set(supervoxels))
    ids_to_read = paired_ids[:max_cases] if max_cases is not None else paired_ids

    if sitk is None and nib is None:
        ids_to_read = []
        warnings.append("Content checks skipped: install SimpleITK or nibabel to read .nii.gz files")

    for case_id in ids_to_read:
        try:
            image_meta, image_array = read_image(images[case_id])
        except Exception as exc:  # noqa: BLE001 - report every unreadable case
            errors.append(f"Cannot read image {images[case_id]}: {exc}")
            continue

        try:
            label_meta, label_array = read_image(labels[case_id])
        except Exception as exc:  # noqa: BLE001 - report every unreadable case
            errors.append(f"Cannot read label {labels[case_id]}: {exc}")
            continue

        if image_array.shape != label_array.shape:
            errors.append(
                f"Shape mismatch for case {case_id}: image {image_array.shape}, label {label_array.shape}"
            )
        if not same_metadata(image_meta, label_meta):
            warnings.append(f"Metadata mismatch for case {case_id}: image and label spacing/origin/direction differ")

        label_values = np.unique(label_array)
        label_value_counts.update(int(value) for value in label_values)
        unexpected_labels = sorted(set(map(int, label_values)) - layout.expected_labels)
        if unexpected_labels:
            message = f"Unexpected label values for case {case_id}: {unexpected_labels}"
            if strict_labels:
                errors.append(message)
            else:
                warnings.append(message)

        if check_supervoxels and layout.supervoxel_dir is not None:
            try:
                supervoxel_meta, supervoxel_array = read_image(supervoxels[case_id])
            except Exception as exc:  # noqa: BLE001 - report every unreadable case
                errors.append(f"Cannot read supervoxel {supervoxels[case_id]}: {exc}")
                continue

            if image_array.shape != supervoxel_array.shape:
                errors.append(
                    f"Shape mismatch for case {case_id}: image {image_array.shape}, "
                    f"supervoxel {supervoxel_array.shape}"
                )
            if not same_metadata(image_meta, supervoxel_meta):
                warnings.append(
                    f"Metadata mismatch for case {case_id}: image and supervoxel spacing/origin/direction differ"
                )
            if np.unique(supervoxel_array).size <= 1:
                warnings.append(f"Supervoxel case {case_id} has <= 1 unique value")

        if image_array.size == 0:
            errors.append(f"Image case {case_id} is empty")
        elif not np.isfinite(image_array).all():
            errors.append(f"Image case {case_id} contains NaN or Inf")
        elif float(np.std(image_array)) == 0.0:
            warnings.append(f"Image case {case_id} has zero intensity variance")

    print(f"\n[{layout.name}]")
    print(f"Images: {len(images)}  Labels: {len(labels)}")
    if check_supervoxels and layout.supervoxel_dir is not None:
        print(f"Supervoxels: {len(supervoxels)}")
    print(f"Paired cases checked: {len(ids_to_read)} / {len(paired_ids)}")
    if label_value_counts:
        print(f"Observed label values: {sorted(label_value_counts)}")

    print_messages("ERROR", errors)
    print_messages("WARN", warnings)

    if not errors and not warnings:
        print("OK: no problems found.")
    elif not errors:
        print("OK with warnings.")
    else:
        print("FAILED.")

    return len(errors)


def compact_ids(ids: list[int], limit: int = 30) -> str:
    shown = ", ".join(map(str, ids[:limit]))
    if len(ids) > limit:
        shown += f", ... (+{len(ids) - limit} more)"
    return shown


def print_messages(level: str, messages: list[str]) -> None:
    for message in messages:
        print(f"{level}: {message}")


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 2

    layouts = choose_layouts(root, args.layout)
    if not layouts:
        print(
            f"ERROR: no known dataset layout found under {root}. "
            "Try --layout cmr, --layout chaost2, or --layout split.",
            file=sys.stderr,
        )
        return 2

    error_count = 0
    for layout in layouts:
        error_count += check_layout(
            layout,
            check_supervoxels=args.check_supervoxels,
            strict_labels=args.strict_labels,
            max_cases=args.max_cases,
        )

    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
