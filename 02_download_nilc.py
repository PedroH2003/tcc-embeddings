import os
import shutil
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

# 1. Definicao das pastas de destino
MODELS_DIR = "models"
W2V_DIR = os.path.join(MODELS_DIR, "word2vec")
FT_DIR = os.path.join(MODELS_DIR, "fasttext")

os.makedirs(W2V_DIR, exist_ok=True)
os.makedirs(FT_DIR, exist_ok=True)

def download_nilc_model(repo_id, target_dir, model_name):
    print(f"\n=================================================================")
    print(f"Baixando {model_name} de: {repo_id}")
    print(f"Destino: {target_dir}")
    print(f"=================================================================")
    
    safetensors_target = os.path.join(target_dir, "embeddings.safetensors")
    vocab_target = os.path.join(target_dir, "vocab.txt")
    
    # 1. Download do arquivo de pesos (embeddings.safetensors ~1.1 GB)
    if not os.path.exists(safetensors_target):
        print("Baixando embeddings.safetensors (aguarde o download)...")
        cached_path = hf_hub_download(repo_id=repo_id, filename="embeddings.safetensors")
        shutil.copy(cached_path, safetensors_target)
        print(f"[OK] embeddings.safetensors salvo em {target_dir}")
    else:
        print(f"[CACHE] embeddings.safetensors ja existe em {target_dir}")
        
    # 2. Download do vocabulario (vocab.txt)
    if not os.path.exists(vocab_target):
        print("Baixando vocab.txt...")
        cached_vocab = hf_hub_download(repo_id=repo_id, filename="vocab.txt")
        shutil.copy(cached_vocab, vocab_target)
        print(f"[OK] vocab.txt salvo em {target_dir}")
    else:
        print(f"[CACHE] vocab.txt ja existe em {target_dir}")
        
    # 3. Teste rápido de integridade
    print("Verificando integridade do arquivo baixado...")
    data = load_file(safetensors_target)
    weights = data["embeddings"]
    with open(vocab_target, "r", encoding="utf-8") as f:
        vocab_size = sum(1 for _ in f)
        
    print(f"[TESTE SUCESSO] Matriz carregada: {weights.shape[0]} palavras x {weights.shape[1]} dimensoes.")
    print(f"[TESTE SUCESSO] Linhas no vocab.txt: {vocab_size}")

# =====================================================================
# EXECUÇÃO DO DOWNLOAD DOS 2 MODELOS ESTÁTICOS DO NILC
# =====================================================================
# 1. Word2Vec Skip-Gram 300d
download_nilc_model(
    repo_id="nilc-nlp/word2vec-skip-gram-300d",
    target_dir=W2V_DIR,
    model_name="Word2Vec (Skip-Gram 300d - NILC/USP)"
)

# 2. FastText Skip-Gram 300d
download_nilc_model(
    repo_id="nilc-nlp/fasttext-skip-gram-300d",
    target_dir=FT_DIR,
    model_name="FastText (Skip-Gram 300d - NILC/USP)"
)

print("\n>>> DOWNLOAD DOS MODELOS DO NILC CONCLUIDO COM SUCESSO!")