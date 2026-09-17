import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Carregar a base inteira de treino
train_path = os.path.join('data', 'case01', 'train_full.csv')
df_train = pd.read_csv(train_path)

cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']

# 2. Extrair amostragem representativa de 100.000 dados com semente fixa
df_sample = df_train[cols_notas].sample(n=100000, random_state=42).reset_index(drop=True)
X_sample = df_sample.values

# 3. Definir intervalo de K a ser testado
k_range = range(2, 11)
inertia_list = []
silhouette_list = []

print(f"Iniciando a avaliação do K-Means na amostra representativa ({X_sample.shape[0]} amostras)...")

for k in k_range:
    print(f"-> Processando K = {k}...")
    
    # Treino do K-Means na amostra
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sample)
    
    # Inércia (SSE / Cotovelo)
    inertia = kmeans.inertia_
    inertia_list.append(inertia)
    
    # Coeficiente de Silhueta Médio
    sil_score = silhouette_score(X_sample, labels, metric='euclidean')
    silhouette_list.append(sil_score)
    
    print(f"   Concluído | Inércia: {inertia:.2f} | Silhueta Média: {sil_score:.4f}")

# 4. Plotagem das Curvas (Cotovelo e Silhueta)
fig, ax1 = plt.subplots(figsize=(10, 5))

# Gráfico da Inércia (Método do Cotovelo)
color = 'tab:blue'
ax1.set_xlabel('Número de Clusters (K)')
ax1.set_ylabel('Inércia / SSE (Cotovelo)', color=color)
ax1.plot(k_range, inertia_list, marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(k_range)
ax1.grid(True, linestyle='--', alpha=0.5)

# Gráfico do Coeficiente de Silhueta Médio
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Coeficiente de Silhueta Médio', color=color)
ax2.plot(k_range, silhouette_list, marker='s', color=color, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Avaliação do K Ideal: Método do Cotovelo vs. Silhueta Média (Amostra N=100.000)')
fig.tight_layout()

# Salvar gráfico na pasta images/case01
output_plot = os.path.join('images', 'case01', 'kmeans_k_evaluation_sample100k.png')
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"\nGráfico gerado e salvo em: {output_plot}")