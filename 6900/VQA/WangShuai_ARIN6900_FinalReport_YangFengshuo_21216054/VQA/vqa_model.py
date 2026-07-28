# coding: utf-8
import torch
from PIL import Image
from transformers import (
    ViltProcessor, ViltForQuestionAnswering,
    BlipProcessor, BlipForQuestionAnswering,
    AutoProcessor, AutoModelForCausalLM
)

class BaseVQAModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def predict_batch(self, images: list[Image.Image], questions: list[str]) -> list[tuple[str, float]]:
        """接收一组图片和一组问题（一一对应），返回 [(答案1, 置信度1), (答案2, 置信度2), ...]"""
        raise NotImplementedError("子类必须实现 predict_batch 方法")


class ViLTInferenceModel(BaseVQAModel):
    """ViLT 批量推理封装"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device)
        print("Loading ViLT Model...", flush=True)
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, images: list[Image.Image], questions: list[str]) -> list[tuple[str, float]]:
        # 统一确保所有图片都是 RGB 格式
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        
        inputs = self.processor(images, questions, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=-1)
        confs = probs.max(dim=-1).values.tolist()
        ans_indices = outputs.logits.argmax(-1).tolist()
        
        results = []
        for idx, conf in zip(ans_indices, confs):
            ans_text = self.model.config.id2label[idx].strip().lower()
            results.append((ans_text, conf))
        return results


class BLIPInferenceModel(BaseVQAModel):
    """BLIP 批量推理封装"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device)
        print("Loading BLIP Model...", flush=True)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        # 生成模型批量处理核心：必须左填充
        self.processor.tokenizer.padding_side = "left"
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, images: list[Image.Image], questions: list[str]) -> list[tuple[str, float]]:
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        inputs = self.processor(images=images, text=questions, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=10, 
                return_dict_in_generate=True, 
                output_scores=True
            )
            
        answers = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        # 提取 Batch 中每个样本第一个 Token 的置信度
        if hasattr(outputs, "scores") and len(outputs.scores) > 0:
            first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
            confs = first_token_probs.max(dim=-1).values.tolist()
        else:
            confs = [1.0] * len(images)
            
        return [(ans.strip().lower(), conf) for ans, conf in zip(answers, confs)]


class GITInferenceModel(BaseVQAModel):
    """GIT 批量推理封装"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device)
        print("Loading GIT Model...", flush=True)
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")
        # 自回归 Causal LM 批量生成核心：必须配置左填充和 Pad Token
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, images: list[Image.Image], questions: list[str]) -> list[tuple[str, float]]:
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        inputs = self.processor(images=images, text=questions, padding=True, return_tensors="pt").to(self.device)
        max_prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=10, 
                return_dict_in_generate=True, 
                output_scores=True
            )
            
        # 动态切片：左填充后，所有新生成的答案 Token 都在 max_prompt_len 之后
        answer_sequences = [seq[max_prompt_len:] for seq in outputs.sequences]
        answers = self.processor.batch_decode(answer_sequences, skip_special_tokens=True)
        
        if hasattr(outputs, "scores") and len(outputs.scores) > 0:
            first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
            confs = first_token_probs.max(dim=-1).values.tolist()
        else:
            confs = [1.0] * len(images)
            
        return [(ans.strip().lower(), conf) for ans, conf in zip(answers, confs)]


def get_vqa_model(model_name: str, device: str = "cuda") -> BaseVQAModel:
    name_lower = model_name.lower()
    if name_lower == "vilt":
        return ViLTInferenceModel(device=device)
    elif name_lower == "blip":
        return BLIPInferenceModel(device=device)
    elif name_lower == "git":
        return GITInferenceModel(device=device)
    else:
        raise ValueError(f" Model Name Error! ")