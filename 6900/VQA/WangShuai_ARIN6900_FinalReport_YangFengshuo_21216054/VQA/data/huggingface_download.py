from huggingface_hub import snapshot_download


models_to_download = [
    "diffusion_pytorch_model.safetensors"
]

def download_all():
    
    for repo_id in models_to_download:
        print(f"\n==================================================")
        print(f"Download: {repo_id}")
        print(f"==================================================")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                max_workers=4,         
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                local_dir_use_symlinks=True 
            )
        except Exception as e:
            print(f"{repo_id} Error: {e}")

if __name__ == "__main__":
    download_all()