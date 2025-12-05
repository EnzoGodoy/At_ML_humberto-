import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Obter o diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'all_stocks_5yr.csv')

print(f"📁 Carregando dados de: {csv_path}")

# Carregar os dados
df = pd.read_csv(csv_path)

# Pré-processamento básico
print("📊 Pré-processando dados...")

# Criar variável target: 1 se fechamento > abertura, 0 caso contrário
df['target'] = np.where(df['close'] > df['open'], 1, 0)

# Selecionar duas features para visualização (abertura e volume normalizado)
# Usar log do volume para melhor visualização
df['log_volume'] = np.log1p(df['volume'])

# Features para visualização 2D
X = df[['open', 'log_volume']].values
y = df['target'].values

# Amostrar aleatoriamente para não sobrecarregar o gráfico
np.random.seed(42)
sample_indices = np.random.choice(len(X), size=1000, replace=False)
X_sample = X[sample_indices]
y_sample = y[sample_indices]

# Normalizar as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Criar o classificador KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y_sample)

# Configurar o mesh grid para plotar o limite de decisão
h = 0.02  # tamanho do passo no mesh
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Prever classes para cada ponto no mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Configurar o plot
plt.figure(figsize=(14, 10))

# Definir cores para as classes
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])  # Vermelho claro para queda, Verde claro para alta
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])   # Vermelho para queda, Verde para alta

# Plotar o limite de decisão
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Plotar os pontos de dados
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_sample, 
                     cmap=cmap_bold, edgecolor='black', s=30, alpha=0.7)

# Configurar título e labels
plt.title("KNN Decision Boundary - Previsão de Alta/Queda de Ações\n(Open Price vs Log Volume)", 
          fontsize=16, fontweight='bold', pad=20)

plt.xlabel("Preço de Abertura (Normalizado)", fontsize=12)
plt.ylabel("Volume (Log Normalizado)", fontsize=12)

# Adicionar grid
plt.grid(True, linestyle='--', alpha=0.3)

# Adicionar legenda
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', 
               markersize=10, label='Queda (Close ≤ Open)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00', 
               markersize=10, label='Alta (Close > Open)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=11)

# Adicionar informações no gráfico
plt.text(0.02, 0.98, f'K = {knn.n_neighbors}\nAmostras: {len(X_sample)}', 
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adicionar anotações para as regiões
plt.text(-2, -2, 'Região de Queda', fontsize=11, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
plt.text(1.5, 1.5, 'Região de Alta', fontsize=11, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

# Ajustar layout
plt.tight_layout()

# Salvar como SVG
output_path = os.path.join(script_dir, 'knn_decision_boundary.svg')
plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
print(f"✅ Gráfico salvo como: {output_path}")

# Mostrar o plot
plt.show()

# Estatísticas adicionais
print(f"\n📊 Estatísticas do modelo:")
print(f"Total de amostras: {len(X_sample)}")
print(f"Quedas: {np.sum(y_sample == 0)}")
print(f"Altas: {np.sum(y_sample == 1)}")
accuracy = knn.score(X_scaled, y_sample)
print(f"Acurácia no conjunto de treino: {accuracy:.3f}")