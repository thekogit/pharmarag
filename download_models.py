import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    print("--- PharmaRAG Model Downloader ---")
    
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Download Octen-Embedding-4B (used for vectorization)
    print("\n[1/3] Checking Octen-Embedding-4B...")
    snapshot_download(
        repo_id="Octen/Octen-Embedding-4B",
        local_dir=os.path.join(models_dir, "Octen-Embedding-4B"),
        local_dir_use_symlinks=False
    )

    # 2. Download mxbai-rerank-base-v2 GGUF (used for precision)
    print("\n[2/3] Checking mxbai-rerank-base-v2 GGUF...")
    reranker_filename = "mxbai-rerank-base-v2.i1-Q4_K_M.gguf"
    reranker_path = os.path.join(models_dir, reranker_filename)
    
    if not os.path.exists(reranker_path):
        hf_hub_download(
            repo_id="mradermacher/mxbai-rerank-base-v2-i1-GGUF",
            filename=reranker_filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded reranker to {reranker_path}")
    else:
        print(f"Reranker already exists at {reranker_path}")

    # 3. Download Default LLM GGUF (Qwen2.5-7B-Instruct)
    print("\n[3/3] Checking Default LLM (Qwen2.5-7B-Instruct) GGUF...")
    llm_filename = "qwen2.5-7b-instruct-q4_k_m.gguf"
    llm_path = os.path.join(models_dir, llm_filename)
    
    if not os.path.exists(llm_path):
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename=llm_filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded LLM to {llm_path}")
    else:
        print(f"LLM already exists at {llm_path}")

    print("\n--- All models verified ---")
    print(f"\nTip: Ensure your .env file has MODEL_PATH={llm_path}")

if __name__ == "__main__":
    download_models()
