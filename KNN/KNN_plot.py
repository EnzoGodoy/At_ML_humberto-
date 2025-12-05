import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier  # ← IMPORT ADICIONADO
import numpy as np
import pandas as pd

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['svg.fonttype'] = 'none'  # Para texto editável no SVG

# Obter diretório do script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'knn_output.pkl')

print("📊 Gerando gráficos...")

if not os.path.exists(data_path):
    print("❌ Arquivo de dados não encontrado! Execute KNN_model.py primeiro.")
    exit()

# Carregar dados
data = joblib.load(data_path)
y_test = data['y_test']
y_pred = data['y_pred']
X_test = data['X_test']
model = data['model']
scaler = data['scaler']
X_train_scaled = data['X_train_scaled']
y_train = data['y_train']

# 1. Gráfico de Confusão (SVG)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Queda', 'Alta'], 
            yticklabels=['Queda', 'Alta'],
            cbar_kws={'label': 'Quantidade'})
plt.title('Matriz de Confusão - Modelo KNN', fontsize=16, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Previsão do Modelo', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'confusion_matrix.svg'), format='svg', bbox_inches='tight')
plt.close()

# 2. Gráfico de Acurácia por Número de Vizinhos (K)
plt.figure(figsize=(12, 6))
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(data['X_test_scaled'])
    accuracies.append(accuracy_score(y_test, y_pred_temp))

plt.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Acurácia vs Número de Vizinhos (K)', fontsize=16, fontweight='bold')
plt.xlabel('Número de Vizinhos (K)', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'k_accuracy.svg'), format='svg', bbox_inches='tight')
plt.close()

# 3. Gráfico de Distribuição das Classes Reais
plt.figure(figsize=(10, 6))
class_counts = pd.Series(y_test).value_counts()
colors = ['#FF6B6B', '#4ECDC4']  # Vermelho para queda, Verde para alta

plt.bar(['Queda (0)', 'Alta (1)'], class_counts.values, color=colors, alpha=0.8)
plt.title('Distribuição das Classes no Conjunto de Teste', fontsize=16, fontweight='bold')
plt.ylabel('Quantidade de Amostras', fontsize=12)
plt.xlabel('Classe', fontsize=12)

# Adicionar valores nas barras
for i, count in enumerate(class_counts.values):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'class_distribution.svg'), format='svg', bbox_inches='tight')
plt.close()

# 4. Gráfico de Métricas por Classe
plt.figure(figsize=(12, 8))
report = classification_report(y_test, y_pred, output_dict=True)

metrics = ['precision', 'recall', 'f1-score']
classes = ['Queda (0)', 'Alta (1)']
values = {metric: [report[str(cls)][metric] for cls in [0, 1]] for metric in metrics}

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, values[metric], width, label=metric.capitalize(), alpha=0.8)

ax.set_title('Métricas de Desempenho por Classe', fontsize=16, fontweight='bold')
ax.set_ylabel('Valor', fontsize=12)
ax.set_xlabel('Classe', fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(classes)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'class_metrics.svg'), format='svg', bbox_inches='tight')
plt.close()

# 5. Gráfico de Correlação entre Features (Heatmap)
plt.figure(figsize=(10, 8))
correlation_matrix = X_test.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, cbar_kws={'label': 'Coeficiente de Correlação'})
plt.title('Matriz de Correlação entre Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'correlation_heatmap.svg'), format='svg', bbox_inches='tight')
plt.close()

print("✅ Gráficos SVG gerados com sucesso!")
print("📁 Arquivos criados:")
print("   - confusion_matrix.svg")
print("   - k_accuracy.svg")
print("   - class_distribution.svg")
print("   - class_metrics.svg")
print("   - correlation_heatmap.svg")