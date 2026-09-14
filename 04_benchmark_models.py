import os
import gc
import re
import time
import psutil
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from safetensors.numpy import load_file
from sentence_transformers import SentenceTransformer

# 1. Garante diretorios de saida
RESULTS_DIR = "results"
PREDS_DIR = os.path.join(RESULTS_DIR, "raw_predictions")
os.makedirs(PREDS_DIR, exist_ok=True)

# 2. Carrega os dados processados
print("Carregando bases padronizadas...")
df_assin = pd.read_csv("data/processed/assin2_test.csv")
df_b2w_train = pd.read_csv("data/processed/b2w_train.csv")
df_b2w_test = pd.read_csv("data/processed/b2w_test.csv")

# Funcao auxiliar para medir memoria RAM do processo atual (em MB)
def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Funcao para calcular cosseno entre duas matrizes de vetores
def compute_cosine_similarity(v1, v2):
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    # Evita divisao por zero caso algum vetor seja nulo
    denom = np.maximum(norm1 * norm2, 1e-9)
    return dot / denom

# =====================================================================
# CLASSE DE CARREGAMENTO DOS MODELOS ESTÁTICOS DO NILC (NumPy puro)
# =====================================================================
class StaticEmbeddingModel:
    def __init__(self, model_dir):
        safetensors_path = os.path.join(model_dir, "embeddings.safetensors")
        vocab_path = os.path.join(model_dir, "vocab.txt")
        
        # Carrega matriz de pesos
        data = load_file(safetensors_path)
        self.vectors = data["embeddings"]
        self.dim = self.vectors.shape[1]
        
        # Mapeia vocabulario palavra -> indice
        self.word2id = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.word2id[line.strip()] = idx

    def encode(self, texts, batch_size=None):
        embeddings = []
        for text in texts:
            # Tokenizacao simples por palavras
            words = re.findall(r'\b\w+\b', str(text).lower())
            word_vectors = [self.vectors[self.word2id[w]] for w in words if w in self.word2id]
            
            if word_vectors:
                # Mean Pooling: media aritmetica dos vetores das palavras da frase
                mean_vec = np.mean(word_vectors, axis=0)
            else:
                # Caso nenhuma palavra exista no vocabulario (vetor zerado)
                mean_vec = np.zeros(self.dim, dtype=np.float32)
            embeddings.append(mean_vec)
        return np.array(embeddings, dtype=np.float32)

# =====================================================================
# FUNÇÃO CENTRAL DE AVALIAÇÃO DE UM MODELO
# =====================================================================
def evaluate_model(model_name, model_obj):
    print(f"\n=================================================================")
    print(f"AVALIANDO: {model_name}")
    print(f"=================================================================")
    
    # -------------------------------------------------------------
    # TASK A: ASSIN 2 (Similaridade Semantica por Cosseno)
    # -------------------------------------------------------------
    print("Executando Task A (ASSIN 2)...")
    ram_before_a = get_ram_usage_mb()
    start_time_a = time.perf_counter()
    
    vecs_a = model_obj.encode(df_assin['text_a'].tolist(), batch_size=32)
    vecs_b = model_obj.encode(df_assin['text_b'].tolist(), batch_size=32)
    
    pred_cosine = compute_cosine_similarity(vecs_a, vecs_b)
    elapsed_time_a = time.perf_counter() - start_time_a
    ram_after_a = get_ram_usage_mb()
    
    # Correlacoes estatisticas com a nota humana
    gold_scores = df_assin['gold_score'].values
    p_corr, _ = pearsonr(pred_cosine, gold_scores)
    s_corr, _ = spearmanr(pred_cosine, gold_scores)
    
    latency_per_pair_ms = (elapsed_time_a / len(df_assin)) * 1000
    
    # Salva palpites da Task A para auditoria
    df_pred_a = df_assin.copy()
    df_pred_a['pred_cosine'] = pred_cosine
    df_pred_a.to_csv(os.path.join(PREDS_DIR, f"task_a_{model_name}.csv"), index=False)
    
    print(f"Task A Concluida: Pearson = {p_corr:.4f} | Spearman = {s_corr:.4f} | Latencia/Par = {latency_per_pair_ms:.2f} ms")
    
    # -------------------------------------------------------------
    # TASK B: B2W-Reviews (Classificacao de Sentimento)
    # -------------------------------------------------------------
    print("Executando Task B (B2W-Reviews: 3000 Treino / 1000 Teste)...")
    ram_before_b = get_ram_usage_mb()
    start_time_b = time.perf_counter()
    
    train_vecs = model_obj.encode(df_b2w_train['text'].tolist(), batch_size=32)
    test_vecs = model_obj.encode(df_b2w_test['text'].tolist(), batch_size=32)
    
    encoding_time_b = time.perf_counter() - start_time_b
    total_texts_b = len(df_b2w_train) + len(df_b2w_test)
    throughput_b = total_texts_b / encoding_time_b
    latency_per_text_ms = (encoding_time_b / total_texts_b) * 1000
    
    # Treina o classificador leve (Regressao Logistica)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_vecs, df_b2w_train['label'].values)
    
    preds_test = clf.predict(test_vecs)
    acc = accuracy_score(df_b2w_test['label'].values, preds_test)
    f1 = f1_score(df_b2w_test['label'].values, preds_test, average='macro')
    ram_after_b = get_ram_usage_mb()
    
    # Salva palpites da Task B para auditoria
    df_pred_b = df_b2w_test.copy()
    df_pred_b['predicted_label'] = preds_test
    df_pred_b.to_csv(os.path.join(PREDS_DIR, f"task_b_{model_name}.csv"), index=False)
    
    print(f"Task B Concluida: Acuracia = {acc*100:.2f}% | Macro F1 = {f1:.4f} | Throughput = {throughput_b:.1f} frases/s")
    
    # Retorna resultados consolidados
    metrics_a = {
        "Model": model_name,
        "Pearson": round(p_corr, 4),
        "Spearman": round(s_corr, 4),
        "Latency_per_pair_ms": round(latency_per_pair_ms, 2),
        "RAM_Peak_MB": round(max(ram_after_a, ram_after_b), 1)
    }
    
    metrics_b = {
        "Model": model_name,
        "Accuracy": round(acc * 100, 2),
        "Macro_F1": round(f1, 4),
        "Throughput_sent_per_sec": round(throughput_b, 1),
        "Latency_per_sent_ms": round(latency_per_text_ms, 2),
        "RAM_Peak_MB": round(max(ram_after_a, ram_after_b), 1)
    }
    
    return metrics_a, metrics_b

# =====================================================================
# EXECUÇÃO SEQUENCIAL DOS 5 MODELOS
# =====================================================================
results_a = []
results_b = []

models_config = [
    ("Word2Vec", "static", "models/word2vec"),
    ("FastText", "static", "models/fasttext"),
    ("MiniLM-L12", "transformer", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("BERTimbau", "transformer", "neuralmind/bert-base-portuguese-cased"),
    ("BGE-M3", "transformer", "BAAI/bge-m3")
]

for name, m_type, path in models_config:
    print(f"\nCarregando pesos de: {name}...")
    
    if m_type == "static":
        model_instance = StaticEmbeddingModel(path)
    else:
        # Transformers usam truncate de 128 tokens para execucao eficiente na CPU
        model_instance = SentenceTransformer(path, device="cpu")
        model_instance.max_seq_length = 128
        
    m_a, m_b = evaluate_model(name, model_instance)
    results_a.append(m_a)
    results_b.append(m_b)
    
    # Salva progresso intermediario (caso a maquina sofra queda de energia, nada se perde)
    pd.DataFrame(results_a).to_csv(os.path.join(RESULTS_DIR, "metrics_task_a.csv"), index=False)
    pd.DataFrame(results_b).to_csv(os.path.join(RESULTS_DIR, "metrics_task_b.csv"), index=False)
    
    # LIMPEZA OBRIGATÓRIA DE MEMÓRIA RAM
    del model_instance
    gc.collect()
    print(f"[MEMÓRIA LIBERADA] RAM limpa com sucesso apos finalizar {name}.")

print("\n=================================================================")
print(">>> BENCHMARK CONCLUÍDO COM SUCESSO! TABELAS FINAIS GERADAS! <<<")
print("=================================================================")