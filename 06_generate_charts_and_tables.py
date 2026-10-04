import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuracao global para normas ABNT / IFTM (Fonte Arial)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Carrega os dados reais do benchmark
df_task_a = pd.read_csv("results/metrics_task_a.csv")
df_task_b = pd.read_csv("results/metrics_task_b.csv")
df_stress = pd.read_csv("results/stress_test_results.csv")
df_pred_bert = pd.read_csv("results/raw_predictions/task_a_BERTimbau.csv")
df_pred_minilm = pd.read_csv("results/raw_predictions/task_a_MiniLM-L12.csv")

# Estilos consistentes por modelo: Cor e Marcador Geometrico
model_styles = {
    "Word2Vec":   {"color": "#555555", "marker": "o"},   # Círculo cinza
    "FastText":   {"color": "#8c564b", "marker": "s"},   # Quadrado marrom
    "MiniLM-L12": {"color": "#2ca02c", "marker": "^"},   # Triângulo verde
    "BERTimbau":  {"color": "#1f77b4", "marker": "D"},   # Losango azul
    "BGE-M3":     {"color": "#6a3d9a", "marker": "h"}    # Hexágono roxo
}

print("Gerando graficos com layout academico padronizado (sem hardcoding)...")

# =====================================================================
# GRÁFICO 1: Trade-Off da Tarefa B (Acurácia vs Throughput na CPU)
# =====================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

for _, row in df_task_b.iterrows():
    m = row["Model"]
    style = model_styles[m]
    ax.scatter(row["Throughput_sent_per_sec"], row["Accuracy"], 
               color=style["color"], marker=style["marker"], 
               s=150, zorder=5, label=m)

ax.set_xscale("log")
ax.set_xlabel("Vazão de Processamento na CPU (frases por segundo - escala logarítmica)", fontsize=10)
ax.set_ylabel("Acurácia na Classificação (%)", fontsize=10)
ax.set_ylim(88, 98.5)
ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

# LEGENDA NO TOPO HORIZONTAL: Impossivel colidir com os pontos
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
          ncol=5, frameon=True, fontsize=9, title="Modelos")

plt.tight_layout()
g1_path = os.path.join(FIG_DIR, "grafico1_tradeoff_classificacao.png")
plt.savefig(g1_path, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 1 padronizado: {g1_path}")

# =====================================================================
# GRÁFICO 2: Trade-Off da Tarefa A (Similaridade Pearson vs Latência)
# =====================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

for _, row in df_task_a.iterrows():
    m = row["Model"]
    style = model_styles[m]
    ax.scatter(row["Latency_per_pair_ms"], row["Pearson"], 
               color=style["color"], marker=style["marker"], 
               s=150, zorder=5, label=m)

ax.set_xlabel("Latência Média por Par de Sentenças na CPU (milissegundos)", fontsize=10)
ax.set_ylabel("Correlação de Pearson (r)", fontsize=10)
ax.set_ylim(0.55, 0.85)
ax.grid(True, ls="--", lw=0.5, alpha=0.5)

# LEGENDA NO TOPO HORIZONTAL
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
          ncol=5, frameon=True, fontsize=9, title="Modelos")

plt.tight_layout()
g2_path = os.path.join(FIG_DIR, "grafico2_tradeoff_similaridade.png")
plt.savefig(g2_path, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 2 padronizado: {g2_path}")

# =====================================================================
# GRÁFICO 3: Diagnóstico dos Testes de Estresse (Barras)
# =====================================================================
pivot_stress = df_stress.pivot(index="Modelo", columns="Teste", values="Cosseno")
pivot_stress = pivot_stress.loc[["Word2Vec", "FastText", "MiniLM-L12", "BERTimbau", "BGE-M3"]]

fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
x = np.arange(len(pivot_stress.index))
width = 0.24

rects1 = ax.bar(x - width, pivot_stress["Sentidos opostos (bom vs NAO bom)"], width, label="Negação (Esperado: Baixo)", color="#e74c3c")
rects2 = ax.bar(x, pivot_stress["Mesmo sentido, mas com erros de digitacao e girias"], width, label="Erro Ortográfico (Esperado: Alto)", color="#2ecc71")
rects3 = ax.bar(x + width, pivot_stress["Banco financeiro vs Instituicao financeira"], width, label="Polissemia (Esperado: Alto)", color="#3498db")

ax.set_ylabel("Similaridade de Cosseno Calculada", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(pivot_stress.index, fontsize=10)
ax.set_ylim(0, 1.15)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, fontsize=9)
ax.grid(axis='y', ls='--', lw=0.5, alpha=0.5)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
g3_path = os.path.join(FIG_DIR, "grafico3_testes_estresse.png")
plt.savefig(g3_path, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 3 padronizado: {g3_path}")

# =====================================================================
# GRÁFICO 4: A Prova Visual da Anisotropia (BERTimbau vs MiniLM)
# =====================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

sns.kdeplot(df_pred_bert['pred_cosine'], ax=ax, label="BERTimbau Base (Espaço Anisotrópico / Cone)", color="#1f77b4", fill=True, alpha=0.35, lw=2)
sns.kdeplot(df_pred_minilm['pred_cosine'], ax=ax, label="MiniLM-L12 (Espaço Isotrópico Calibrado SBERT)", color="#2ca02c", fill=True, alpha=0.35, lw=2)

ax.set_xlabel("Similaridade de Cosseno Prevista", fontsize=10)
ax.set_ylabel("Densidade de Probabilidade", fontsize=10)
ax.set_xlim(-0.1, 1.05)
ax.grid(True, ls="--", lw=0.5, alpha=0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True, fontsize=9)

plt.tight_layout()
g4_path = os.path.join(FIG_DIR, "grafico4_anisotropia_bert_vs_minilm.png")
plt.savefig(g4_path, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 4 padronizado: {g4_path}")

print("\n>>> GRÁFICOS REGENERADOS COM SUCESSO (LAYOUT PROFISSIONAL)! <<<")