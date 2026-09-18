import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Carregar a amostra específica para o Hierárquico (N=10.000)
sample_path = os.path.join('data', 'case01', 'train_sample_hierarchical.csv')
df_sample = pd.read_csv(sample_path)

cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']
X_sample = df_sample[cols_notas].values

# 2. Gerar e plotar os Dendrogramas para 2 critérios de ligação (Ward e Complete)
linkage_methods = ['ward', 'complete']

for method in linkage_methods:
    plt.figure(figsize=(12, 6))
    
    # Matriz de ligação
    Z = linkage(X_sample, method=method)
    
    # Plot do Dendrograma (truncado para legibilidade)
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=30,  # Mostra apenas os últimos 30 clusters unificados
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )
    
    plt.title(f'Dendrograma - Agrupamento Hierárquico Aglomerativo (Ligação: {method.capitalize()})')
    plt.xlabel('Tamanho do Cluster ou Índice do Ponto')
    plt.ylabel('Distância de Agrupamento')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_dendro = os.path.join('images', 'case01', f'dendrogram_{method}.png')
    plt.savefig(output_dendro, dpi=300)
    plt.show()
    print(f"Dendrograma ({method}) salvo em: {output_dendro}")

# ==========================================================================================
# 3. Teste de cortes e avaliação da Silhueta para os dois métodos
# ==========================================================================================
print(f"\n{'='*20} AVALIAÇÃO DO AGRUPAMENTO HIERÁRQUICO {'='*20}")
print(f"{'Método':<12}{'Nº Clusters (K)':<18}{'Silhueta Média':<15}")

for method in linkage_methods:
    for k in range(2, 6):
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage=method)
        labels = model.fit_predict(X_sample)
        
        sil = silhouette_score(X_sample, labels)
        print(f"{method:<12}{k:<18}{sil:.4f}")