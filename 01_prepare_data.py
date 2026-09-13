import os
import urllib.request
import pandas as pd
from datasets import load_dataset

# 1. Pastas do projeto
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =====================================================================
# TASK A: ASSIN 2 (Similaridade)
# =====================================================================
print("--- [1/2] Task A: ASSIN 2 ---")
raw_assin_path = "data/raw/assin2_raw_test.csv"
processed_assin_path = "data/processed/assin2_test.csv"

if not os.path.exists(raw_assin_path):
    print("Baixando dados brutos do ASSIN 2...")
    dataset_assin = load_dataset("nilc-nlp/assin2")
    df_assin_raw = pd.DataFrame(dataset_assin['test'])
    df_assin_raw.to_csv(raw_assin_path, index=False)
else:
    print(f"[CACHE] {raw_assin_path} ja existe.")

df_assin = pd.read_csv(raw_assin_path)

col_a = 'premise' if 'premise' in df_assin.columns else 'sentence1'
col_b = 'hypothesis' if 'hypothesis' in df_assin.columns else 'sentence2'
col_score = 'relatedness_score' if 'relatedness_score' in df_assin.columns else 'similarity_score'

df_assin_clean = df_assin[[col_a, col_b, col_score]].copy()
df_assin_clean.rename(columns={col_a: 'text_a', col_b: 'text_b', col_score: 'gold_score'}, inplace=True)
df_assin_clean.to_csv(processed_assin_path, index=False)
print(f"[OK] Task A salva: {len(df_assin_clean)} pares.")

# =====================================================================
# TASK B: B2W-Reviews (Sentimento)
# =====================================================================
print("\n--- [2/2] Task B: B2W-Reviews ---")
raw_b2w_path = "data/raw/b2w_raw.csv"
url_b2w = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv"

if not os.path.exists(raw_b2w_path):
    print("Baixando dados brutos da B2W...")
    urllib.request.urlretrieve(url_b2w, raw_b2w_path)
else:
    print(f"[CACHE] {raw_b2w_path} ja existe.")

# Leitura padrao com virgula (identica a Task A)
df_raw = pd.read_csv(raw_b2w_path, low_memory=False)

# Filtro basico de textos validos
df_b2w = df_raw[['review_text', 'overall_rating']].dropna()
df_b2w = df_b2w[df_b2w['review_text'].str.strip() != ""]

# Binarizacao: 1 e 2 = Negativo (0) | 4 e 5 = Positivo (1)
df_neg = df_b2w[df_b2w['overall_rating'].isin([1, 2])].sample(n=2000, random_state=42)
df_neg['label'] = 0

df_pos = df_b2w[df_b2w['overall_rating'].isin([4, 5])].sample(n=2000, random_state=42)
df_pos['label'] = 1

# Divisao simples e direta:
# 1.500 de cada para treino (Total: 3.000)
df_train = pd.concat([df_neg.iloc[:1500], df_pos.iloc[:1500]]).sample(frac=1, random_state=42)
df_train = df_train[['review_text', 'label']].rename(columns={'review_text': 'text'})

# 500 restantes de cada para teste (Total: 1.000)
df_test = pd.concat([df_neg.iloc[1500:2000], df_pos.iloc[1500:2000]]).sample(frac=1, random_state=42)
df_test = df_test[['review_text', 'label']].rename(columns={'review_text': 'text'})

df_train.to_csv("data/processed/b2w_train.csv", index=False)
df_test.to_csv("data/processed/b2w_test.csv", index=False)

print(f"[OK] Task B Treino salvo: {len(df_train)} resenhas.")
print(f"[OK] Task B Teste salvo:  {len(df_test)} resenhas.")
print("\n>>> ETAPA DE DADOS CONCLUIDA COM SUCESSO!")