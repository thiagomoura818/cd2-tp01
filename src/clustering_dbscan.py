import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. Carregar a base de treino completa
train_path = os.path.join('data', 'case01', 'train_full.csv')
df_train = pd.read_csv(train_path)

cols_notas = ['NOTA_L', 'NOTA_CH', 'NOTA_CN', 'NOTA_M', 'NOTA_R']

# 2. Amostragem de 100.000 registros para viabilizar o cálculo do k-dist
df_sample = df_train[cols_notas].sample(n=100000, random_state=42).reset_index(drop=True)
X_sample = df_sample.values

# 3. Configuração do MinPts conforme o enunciado: MinPts = 2 * D = 10
D = len(cols_notas)
min_pts = 2 * D  # k = 10

# 4. Cálculo das distâncias ao 10º vizinho mais próximo
nn = NearestNeighbors(n_neighbors=min_pts, n_jobs=-1)
nn.fit(X_sample)
distancias, _ = nn.kneighbors(X_sample)

# Ordenar as distâncias para o k-ésimo vizinho (índice k-1 = 9)
k_dist = np.sort(distancias[:, min_pts - 1])

# 5. Plotagem do Gráfico k-dist
plt.figure(figsize=(9, 5))
plt.plot(k_dist, color='#003366', linewidth=1.5)
plt.xlabel('Pontos Ordenados por Distância')
plt.ylabel(f'Distância ao {min_pts}º Vizinho Mais Próximo (eps)')
plt.title(f'Gráfico k-dist (k = {min_pts}) para Escolha do Raio eps Ideal')
plt.grid(True, linestyle='--', alpha=0.5)

# Linha guia recomendada para identificar o "cotovelo" da curva k-dist
# Ajuste o valor de eps_sugerido após observar o primeiro plot gerado
eps_sugerido = 0.8  # Exemplo visual
plt.axhline(eps_sugerido, color='red', linestyle='--', label=f'eps sugerido (~{eps_sugerido})')
plt.legend()

output_plot = os.path.join('images', 'case01', 'dbscan_kdist_graph.png')
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"Gráfico k-dist salvo em: {output_plot}")

# ==========================================================================================
# 6. Testando valores de eps em torno da inflexão do gráfico k-dist
# ==========================================================================================
print(f"\n{'='*20} AVALIAÇÃO DO DBSCAN COM DIFERENTES VALORES DE EPS {'='*20}")
print(f"{'eps':<10}{'n_clusters':<15}{'n_ruido':<15}{'silhueta':<15}")

# Faixa de teste para o raio eps em torno da dobra
faixa_eps = [0.5, 0.7, 0.8, 1.0, 1.2]

for eps in faixa_eps:
    db = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    labels_db = db.fit_predict(X_sample)
    
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_ruido = (labels_db == -1).sum()
    
    # Silhueta calculada apenas se houver mais de 1 cluster e nem tudo for ruído
    if n_clusters > 1 and (len(labels_db) - n_ruido) > 0:
        # Avalia a silhueta excluindo o ruído (-1)
        mask = labels_db != -1
        sil = silhouette_score(X_sample[mask], labels_db[mask], metric='euclidean')
        sil_str = f"{sil:.4f}"
    else:
        sil_str = "N/A"
        
    print(f"{eps:<10}{n_clusters:<15}{n_ruido:<15}{sil_str:<15}")