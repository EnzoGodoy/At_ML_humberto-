import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Carregar os dados
df = pd.read_csv('all_stocks_5yr.csv')

# Filtrar apenas a ação desejada (ex: AAL)
df = df[df['Name'] == 'AAL'].copy()

# Criar a variável target: 1 se o preço sobe no próximo dia, 0 caso contrário
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()  # Remover a última linha (que não tem target)

# Selecionar features (variáveis independentes)
features = ['open', 'high', 'low', 'close', 'volume']
X = df[features]
y = df['target']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados (importante para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = knn.predict(X_test_scaled)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Criar figura com subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Gráfico de Preços com Previsões
axes[0, 0].plot(df.index[-len(y_test):], df['close'].iloc[-len(y_test):], label='Preço Real', alpha=0.7)
axes[0, 0].set_title('Preços de Fechamento (Teste)')
axes[0, 0].set_xlabel('Índice')
axes[0, 0].set_ylabel('Preço')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Matriz de Confusão')
axes[0, 1].set_xlabel('Predito')
axes[0, 1].set_ylabel('Real')

# 3. Distribuição das Previsões
pred_counts = pd.Series(y_pred).value_counts()
axes[1, 0].bar(['Queda (0)', 'Alta (1)'], pred_counts.values, color=['red', 'green'], alpha=0.7)
axes[1, 0].set_title('Distribuição das Previsões')
axes[1, 0].set_ylabel('Quantidade')

# 4. Comparação Real vs Predito (apenas primeiros 50 pontos para clareza)
sample_size = min(50, len(y_test))
axes[1, 1].plot(range(sample_size), y_test.values[:sample_size], 'o-', label='Real', alpha=0.7)
axes[1, 1].plot(range(sample_size), y_pred[:sample_size], 's-', label='Predito', alpha=0.7)
axes[1, 1].set_title('Comparação Real vs Predito (amostra)')
axes[1, 1].set_xlabel('Amostra')
axes[1, 1].set_ylabel('Classe (0=Queda, 1=Alta)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Resultados do KNN para AAL - Acurácia: {accuracy:.2%}', fontsize=16)
plt.tight_layout()

# Salvar como SVG
plt.savefig('knn_results_aal.svg', format='svg')
plt.savefig('knn_results_aal.png', dpi=300)  # Também salva como PNG para referência
plt.close()

print("\nGráfico salvo como 'knn_results_aal.svg'")