
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)
import matplotlib.pyplot as plt


RANDOM_STATE = 42
K_MIN = 2
K_MAX = 8  


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'KNN', 'all_stocks_5yr.csv')
output_dir = os.path.join(script_dir, 'kmeans_evaluation_outputs')
os.makedirs(output_dir, exist_ok=True)


if not os.path.exists(csv_path):
    print('\nErro: arquivo CSV não encontrado em:', csv_path)
    print('Por favor ajuste a variável csv_path no topo do script para apontar ao CSV correto.')
    sys.exit(1)

print('Carregando dados de:', csv_path)
df = pd.read_csv(csv_path)


if 'close' not in df.columns or 'Name' not in df.columns:
    print("Erro: CSV precisa conter as colunas 'close' e 'Name'. Verifique o arquivo.")
    sys.exit(1)


df = df.sort_values(['Name'])
df['return'] = df.groupby('Name')['close'].pct_change()


stats = df.groupby('Name').agg({'return': ['mean', 'std']}).reset_index()
stats.columns = ['Name', 'mean_return', 'volatility']
stats = stats.dropna().reset_index(drop=True)

if stats.shape[0] == 0:
    print('Nenhum dado após agregação (stats vazio). Verifique o CSV e os dados de retorno).')
    sys.exit(1)


X = stats[['mean_return', 'volatility']].values


Ks = list(range(K_MIN, K_MAX + 1))
inertias = []
silhouettes = []
calinski = []
davies = []

print('\nCalculando métricas por K...')
for K in Ks:
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
   
    silhouettes.append(float(silhouette_score(X, labels)))
    calinski.append(float(calinski_harabasz_score(X, labels)))
    davies.append(float(davies_bouldin_score(X, labels)))
    print(f' K={K}: inertia={kmeans.inertia_:.3f}, silhouette={silhouettes[-1]:.3f}, calinski={calinski[-1]:.3f}, davies={davies[-1]:.3f}')

metrics_df = pd.DataFrame({
    'K': Ks,
    'inertia': inertias,
    'silhouette_mean': silhouettes,
    'calinski_harabasz': calinski,
    'davies_bouldin': davies
})
metrics_csv = os.path.join(output_dir, 'kmeans_metrics_by_k.csv')
metrics_df.to_csv(metrics_csv, index=False)
print('\nMétricas por K salvas em:', metrics_csv)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(Ks, inertias, marker='o')
plt.title('Elbow (Inertia) vs K')
plt.xlabel('K')
plt.ylabel('Inertia (WCSS)')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(Ks, silhouettes, marker='o')
plt.title('Silhouette (mean) vs K')
plt.xlabel('K')
plt.ylabel('Silhouette score')
plt.grid(alpha=0.3)
plt.tight_layout()
plot_elbow = os.path.join(output_dir, 'elbow_silhouette_by_k.png')
plt.savefig(plot_elbow, dpi=200, bbox_inches='tight')
plt.close()
print('Plot Elbow/Silhouette salvo em:', plot_elbow)


K_chosen = 3
print('\nExecutando KMeans final com K=', K_chosen)
kmeans = KMeans(n_clusters=K_chosen, random_state=RANDOM_STATE, n_init=10)
labels = kmeans.fit_predict(X)
stats['cluster'] = labels


sil_mean = float(silhouette_score(X, labels))
cal = float(calinski_harabasz_score(X, labels))
dav = float(davies_bouldin_score(X, labels))
inertia = float(kmeans.inertia_)

summary = {
    'K': K_chosen,
    'inertia': inertia,
    'silhouette_mean': sil_mean,
    'calinski_harabasz': cal,
    'davies_bouldin': dav
}
summary_csv = os.path.join(output_dir, 'kmeans_summary.csv')
pd.DataFrame([summary]).to_csv(summary_csv, index=False)
print('Resumo das métricas do K escolhido salvo em:', summary_csv)


clusters_csv = os.path.join(output_dir, 'clusters_result.csv')
stats.to_csv(clusters_csv, index=False)
cluster_summary = stats.groupby('cluster').agg({
    'Name': 'count',
    'volatility': 'mean',
    'mean_return': 'mean'
}).rename(columns={'Name': 'count'}).reset_index()
cluster_summary_csv = os.path.join(output_dir, 'cluster_summary.csv')
cluster_summary.to_csv(cluster_summary_csv, index=False)
print('Resultados dos clusters e resumo salvos em:', clusters_csv, 'e', cluster_summary_csv)


plt.figure(figsize=(8, 6))

scatter = plt.scatter(stats['volatility'], stats['mean_return'], c=stats['cluster'], cmap='viridis', s=40, edgecolor='k', linewidth=0.3)
plt.xlabel('Volatilidade (std dos retornos)')
plt.ylabel('Retorno Médio')
plt.title(f'Clusters de Ações - KMeans (K={K_chosen})')
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 1], centroids[:, 0], marker='X', s=200, c='red', label='Centroids', edgecolor='k')
plt.legend(loc='best')
plt.grid(alpha=0.25, linestyle='--')
plot_scatter = os.path.join(output_dir, 'clusters_scatter_with_centroids.png')
plt.savefig(plot_scatter, dpi=200, bbox_inches='tight')
plt.close()
print('Scatter com centroides salvo em:', plot_scatter)


sil_samples = silhouette_samples(X, labels)

plt.figure(figsize=(8, 6))
y_lower = 10
for i in range(K_chosen):
    ith_sil_values = sil_samples[labels == i]
    ith_sil_values.sort()
    size_cluster_i = ith_sil_values.shape[0]
    if size_cluster_i == 0:
        continue
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil_values, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i} ({size_cluster_i})')
    y_lower = y_upper + 10  

plt.xlabel('Silhouette coefficient values')
plt.ylabel('Cluster')
plt.title(f'Silhouette plot for K={K_chosen}, mean={sil_mean:.3f}')
plot_sil = os.path.join(output_dir, 'silhouette_plot.png')
plt.savefig(plot_sil, dpi=200, bbox_inches='tight')
plt.close()
print('Silhouette plot salvo em:', plot_sil)

print('\nExecução finalizada. Verifique a pasta de saída para CSVs e imagens:\n', output_dir)
print('Sugestão: analise kmeans_metrics_by_k.csv para decidir se K=3 é o melhor K para seu problema.')
