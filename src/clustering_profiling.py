import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# 1. Carregar a base de dados pré-processada (amostra de treino do hierárquico)
train_path = os.path.join('data', 'case01', 'train_sample_hierarchical.csv')
df_sample = pd.read_csv(train_path)

cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']

# 2. Treinar o modelo de melhor desempenho (Hierárquico Complete K=3)
X_scaled = df_sample[cols_notas].values
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
df_sample['CLUSTER'] = model.fit_predict(X_scaled)

# 3. Carregar a base original sem padronização para calcular as médias reais
data_path = os.path.join('data', 'raw', 'chamada_regular_sisu_2023_1.csv')
df_original = pd.read_csv(data_path, encoding='latin-1', sep='|', low_memory=False)

# Converter tipos na base original
for col in ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M']:
    df_original[col] = df_original[col].astype(str).str.replace(',', '.')
    df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
df_original['NOTA_R'] = pd.to_numeric(df_original['NOTA_R'], errors='coerce')

# Recuperar dados originais e unir com as colunas categóricas salvas
df_profile = df_original.loc[df_sample.index, cols_notas].copy()
df_profile['NOME_CURSO'] = df_sample['NOME_CURSO'].values
df_profile['AREA_CONHECIMENTO'] = df_sample['AREA_CONHECIMENTO'].values
df_profile['CLUSTER'] = df_sample['CLUSTER'].values

# ==========================================================================================
# A. TABELA DE MÉDIAS DAS NOTAS NAS ESCALAS ORIGINAIS POR CLUSTER
# ==========================================================================================
print(f"\n{'='*20} TABELA DE MÉDIAS (NOTAS ORIGINAIS) POR CLUSTER {'='*20}")
tabela_medias = df_profile.groupby('CLUSTER')[cols_notas].mean().round(2)
print(tabela_medias)

tabela_medias.to_csv(os.path.join('data', 'case01', 'perfil_medias_clusters.csv'))

# ==========================================================================================
# B. CRUZAMENTO DOS CLUSTERS COM AS ÁREAS DO CONHECIMENTO (PERCENTUAL)
# ==========================================================================================
print(f"\n{'='*20} DISTRIBUIÇÃO POR ÁREA DO CONHECIMENTO (%) {'='*20}")
cruzamento_areas = pd.crosstab(
    df_profile['CLUSTER'], 
    df_profile['AREA_CONHECIMENTO'], 
    normalize='index'
) * 100

print(cruzamento_areas.round(2).astype(str) + '%')
cruzamento_areas.to_csv(os.path.join('data', 'case01', 'perfil_cruzamento_areas.csv'))

# ==========================================================================================
# C. TOP 5 CURSOS E NOMES DAS PERSONAS POR CLUSTER
# ==========================================================================================
print(f"\n{'='*20} DETALHAMENTO DE PERSONAS E TOP CURSOS {'='*20}")

for cluster_id in range(3):
    df_c = df_profile[df_profile['CLUSTER'] == cluster_id]
    top_cursos = df_c['NOME_CURSO'].value_counts().head(5)
    area_predominante = df_c['AREA_CONHECIMENTO'].mode()[0]
    
    print(f"\n--- CLUSTER {cluster_id} ---")
    print(f"Área Predominante Escolhida: {area_predominante}")
    print("Médias das Notas nas Matérias:")
    print(tabela_medias.loc[cluster_id])
    print("\nTop 5 Cursos Mais Procurados:")
    print(top_cursos)