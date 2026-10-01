import os
import gc
import re
import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from sentence_transformers import SentenceTransformer

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Classe para os modelos estaticos (Word2Vec e FastText)
class StaticEmbeddingModel:
    def __init__(self, model_dir):
        safetensors_path = os.path.join(model_dir, "embeddings.safetensors")
        vocab_path = os.path.join(model_dir, "vocab.txt")
        data = load_file(safetensors_path)
        self.vectors = data["embeddings"]
        self.dim = self.vectors.shape[1]
        self.word2id = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.word2id[line.strip()] = idx

    def encode_one(self, text):
        words = re.findall(r'\b\w+\b', str(text).lower())
        word_vectors = [self.vectors[self.word2id[w]] for w in words if w in self.word2id]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        return np.zeros(self.dim, dtype=np.float32)

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

# 2. As 4 Baterias de Testes de Estresse
stress_cases = [
    {
        "category": "1. Negacao",
        "description": "Sentidos opostos (bom vs NAO bom)",
        "sent1": "O produto e muito bom e recomendo a compra.",
        "sent2": "O produto nao e bom e nao recomendo a compra.",
        "expected": "Baixa similaridade (sentidos opostos)"
    },
    {
        "category": "2. Erro de Digitacao / OOV",
        "description": "Mesmo sentido, mas com erros de digitacao e girias",
        "sent1": "O celular veio quebrado e a tela esta trincada.",
        "sent2": "O celulart veio quebraado e a tela ta trincadassa.",
        "expected": "Alta similaridade (mesmo significado real)"
    },
    {
        "category": "3. Ordem das Palavras",
        "description": "Mesmas palavras, sujeitos e objetos invertidos",
        "sent1": "O cliente enganou a loja no pagamento.",
        "sent2": "A loja enganou o cliente no pagamento.",
        "expected": "Similaridade moderada/baixa (papeis semanticos diferentes)"
    },
    {
        "category": "4. Polissemia (Contexto)",
        "description": "Banco financeiro vs Instituicao financeira",
        "sent1": "Preciso ir ao banco pagar a conta atrasada.",
        "sent2": "A instituicao financeira aprovou o meu pagamento.",
        "expected": "Alta similaridade (mesmo contexto financeiro)"
    },
    {
        "category": "4. Polissemia (Conflito)",
        "description": "Banco de sentar vs Instituicao financeira",
        "sent1": "Sentei no banco de madeira da praca publica.",
        "sent2": "A instituicao financeira aprovou o meu pagamento.",
        "expected": "Baixa similaridade (contextos totalmente distintos)"
    }
]

# 3. Execucao dos 5 Modelos
models_config = [
    ("Word2Vec", "static", "models/word2vec"),
    ("FastText", "static", "models/fasttext"),
    ("MiniLM-L12", "transformer", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("BERTimbau", "transformer", "neuralmind/bert-base-portuguese-cased"),
    ("BGE-M3", "transformer", "BAAI/bge-m3")
]

results_table = []

for name, m_type, path in models_config:
    print(f"Executando Teste de Estresse para: {name}...")
    
    if m_type == "static":
        model = StaticEmbeddingModel(path)
        encode_fn = model.encode_one
    else:
        model = SentenceTransformer(path, device="cpu")
        model.max_seq_length = 128
        encode_fn = lambda t: model.encode(t, convert_to_numpy=True)
    
    for case in stress_cases:
        v1 = encode_fn(case["sent1"])
        v2 = encode_fn(case["sent2"])
        sim = cosine_similarity(v1, v2)
        
        results_table.append({
            "Modelo": name,
            "Categoria": case["category"],
            "Teste": case["description"],
            "Cosseno": round(sim, 4),
            "Esperado": case["expected"]
        })
        
    del model
    gc.collect()

# 4. Formata e salva resultados
df_results = pd.DataFrame(results_table)
output_path = os.path.join(RESULTS_DIR, "stress_test_results.csv")
df_results.to_csv(output_path, index=False)

print("\n" + "="*80)
print(">>> RESULTADOS DO TESTE DE ESTRESSE SEMÂNTICO (COSSENO CALCULADO) <<<")
print("="*80)

# Exibe tabela dinamica organizada por Teste
pivot = df_results.pivot(index=["Categoria", "Teste", "Esperado"], columns="Modelo", values="Cosseno")
# Reordena colunas na ordem historica
pivot = pivot[["Word2Vec", "FastText", "MiniLM-L12", "BERTimbau", "BGE-M3"]]
print(pivot.to_string())

print(f"\n[OK] Resultados detalhados salvos em: {output_path}")