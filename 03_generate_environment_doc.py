import os
import sys
import json
import platform
import psutil
import torch
import transformers
import sentence_transformers
import sklearn
import scipy
import pandas
import numpy

# 1. Identificação precisa do Processador no Windows
def get_cpu_name():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        return name.strip()
    except Exception:
        return platform.processor()

cpu_name = get_cpu_name()
physical_cores = psutil.cpu_count(logical=False)
total_cores = psutil.cpu_count(logical=True)
ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
os_info = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"

# 2. Versões exatas dos pacotes instalados
python_ver = sys.version.split()[0]
torch_ver = torch.__version__
transformers_ver = transformers.__version__
st_ver = sentence_transformers.__version__
sklearn_ver = sklearn.__version__
scipy_ver = scipy.__version__
pandas_ver = pandas.__version__
numpy_ver = numpy.__version__

# 3. Salva JSON técnico para auditoria em results/
os.makedirs("results", exist_ok=True)
specs = {
    "hardware": {
        "os": os_info,
        "cpu": cpu_name,
        "physical_cores": physical_cores,
        "logical_cores": total_cores,
        "total_ram_gb": ram_gb
    },
    "software": {
        "python": python_ver,
        "torch_cpu": torch_ver,
        "transformers": transformers_ver,
        "sentence_transformers": st_ver,
        "scikit_learn": sklearn_ver,
        "scipy": scipy_ver,
        "pandas": pandas_ver,
        "numpy": numpy_ver
    }
}

json_path = os.path.join("results", "hardware_specs.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(specs, f, indent=4)

# 4. Redação formal para o Capítulo 3 (Metodologia) do TCC
doc_content = f"""================================================================================
TEXTO PARA O CAPÍTULO DE METODOLOGIA DO TCC: AMBIENTE EXPERIMENTAL E REPRODUTIBILIDADE
(Copie e cole a seção abaixo no seu Microsoft Word)
================================================================================

3.X AMBIENTE EXPERIMENTAL E PROCEDIMENTOS DE REPRODUTIBILIDADE

Com o propósito de assegurar a estrita reprodutibilidade científica dos experimentos e viabilizar uma análise fidedigna do trade-off computacional (alinhada aos preceitos de Green AI), todas as avaliações empíricas foram executadas exclusivamente em Unidade Central de Processamento (CPU), abstendo-se deliberadamente do uso de aceleradores gráficos dedicados (GPUs). Essa restrição metodológica visa simular a realidade de infraestruturas operacionais e servidores de produção de baixo custo computacional.

Para o isolamento das dependências de software e mitigação de inconsistências de versões, estruturou-se um ambiente virtual dedicado em nível de sistema operacional via venv, operando sob controle estrito de versões e fixação de sementes pseudoaleatórias (random seed = 42) para todos os procedimentos estocásticos de amostragem e partição dos dados.

Quadro 1 – Especificações da Infraestrutura de Hardware Utilizada
--------------------------------------------------------------------------------
Componente                    Especificação / Detalhe Técnico
--------------------------------------------------------------------------------
Sistema Operacional           {os_info}
Processador (CPU)             {cpu_name}
Arquitetura de Núcleos        {physical_cores} núcleos físicos ({total_cores} threads lógicas)
Memória RAM Total             {ram_gb} GB
Dispositivo de Aceleração     Nenhum (Execução estrita em CPU)
--------------------------------------------------------------------------------
Fonte: Elaborado pelo autor (2026).

Quadro 2 – Ecossistema de Software e Bibliotecas Científicas
--------------------------------------------------------------------------------
Ferramenta / Pacote           Versão                Papel Metodológico no Trabalho
--------------------------------------------------------------------------------
Python                        {python_ver}                Interpretador base do ambiente
PyTorch (CPU-only)            {torch_ver}          Motor de tensores e redes neurais profundas
Transformers                  {transformers_ver}             Estrutura de execução do modelo BERTimbau
Sentence-Transformers         {st_ver}              Geração de embeddings para sentenças (MiniLM e BGE-M3)
Scikit-Learn                  {sklearn_ver}              Classificador linear (Regressão Logística) e métricas
SciPy                         {scipy_ver}              Cálculo estatístico de correlações (Pearson e Spearman)
Pandas                        {pandas_ver}                Manipulação e partição tabular das bases
NumPy                         {numpy_ver}                Operações matriciais e agregação por Mean Pooling
--------------------------------------------------------------------------------
Fonte: Elaborado pelo autor (2026).

3.X.1 Estrutura de Armazenamento e Auditoria dos Dados

O repositório experimental foi compartimentado em dois níveis hierárquicos de persistência: (i) dados brutos imutáveis (armazenados em data/raw/), que preservam a integridade dos corpora originais obtidos das fontes públicas ASSIN 2 e B2W-Reviews; e (ii) dados processados (em data/processed/), contendo as partições balanceadas utilizadas no benchmark. As representações vetoriais estáticas de 300 dimensões (Word2Vec e FastText) foram incorporadas a partir das distribuições oficiais do NILC/USP sob o algoritmo Skip-Gram, enquanto os modelos contextuais da família Transformer operaram em modo de extração de atributos (feature extraction), preservando seus pesos internos congelados.
================================================================================
"""

output_doc_path = "METHODOLOGY_ENVIRONMENT.txt"
with open(output_doc_path, "w", encoding="utf-8") as f:
    f.write(doc_content)

print(doc_content)
print(f"\n[OK] Documento de Metodologia salvo em: {output_doc_path}")
print(f"[OK] Ficha técnica em JSON salva em: {json_path}")