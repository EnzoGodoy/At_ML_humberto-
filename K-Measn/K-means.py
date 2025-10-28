
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))   # pasta onde está este script
csv_path = os.path.join(script_dir, "..", "KNN", "all_stocks_5yr.csv")  # ../KNN/all_stocks_5yr.csv
output_svg = os.path.join(script_dir, "kmeans_clusters.svg")
output_csv = os.path.join(script_dir, "clusters_result.csv")


print(f"Carregando dados de: {csv_path}")
df = pd.read_csv(csv_path)

df['return'] = df.groupby('Name')['close'].pct_change()


stats = df.groupby('Name').agg({'return': ['mean', 'std']}).reset_index()
stats.columns = ['Name', 'mean_return', 'volatility']


stats = stats.dropna().reset_index(drop=True)
print(f"Número de tickers depois do dropna: {len(stats)}")


X = stats[['mean_return', 'volatility']].values

K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
stats['cluster'] = labels


plt.figure(figsize=(10, 6))
scatter = plt.scatter(stats['volatility'], stats['mean_return'], c=stats['cluster'], cmap='viridis', s=40, edgecolor='k', linewidth=0.3)

plt.xlabel("Volatilidade (std dos retornos)")
plt.ylabel("Retorno Médio")
plt.title(f"Clusters de Ações - KMeans (K={K})")
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')


centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 1], centroids[:, 0], marker='X', s=200, c='red', label='Centroids', edgecolor='k')

plt.legend(loc='best')
plt.grid(alpha=0.25, linestyle='--')

plt.savefig(output_svg, format='svg', bbox_inches='tight', dpi=300)
print(f"Gráfico salvo como: {output_svg}")

plt.show()

stats.to_csv(output_csv, index=False)
print(f"Resultados dos clusters salvos em: {output_csv}")


print("\nResumo por cluster (count, mean volatility, mean return):")
summary = stats.groupby('cluster').agg({
    'Name': 'count',
    'volatility': 'mean',
    'mean_return': 'mean'
}).rename(columns={'Name': 'count'}).reset_index()
print(summary.to_string(index=False))
