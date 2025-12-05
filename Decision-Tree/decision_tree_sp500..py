import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Baixar dataset (só se não existir)
print("Baixando dataset, se necessário...")
DATASET_PATH = kagglehub.dataset_download("camnugent/sandp500")
csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado no dataset.")
file_path = os.path.join(DATASET_PATH, csv_files[0])
print(f"Usando arquivo: {file_path}")

# Carregar dados
df = pd.read_csv(file_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Ordenação
symbol_col = 'Name' if 'Name' in df.columns else None
df = df.sort_values([symbol_col, 'date']) if symbol_col else df.sort_values('date')
df = df.dropna()

# Criar features
df['Return_1d'] = df['close'].pct_change()
df['MA_5'] = df['close'].rolling(5).mean()
df['MA_10'] = df['close'].rolling(10).mean()
df['Volatility_10'] = df['close'].pct_change().rolling(10).std()
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

# Dados de treino e teste
features = ['Return_1d', 'MA_5', 'MA_10', 'Volatility_10']
X, y = df[features], df['Target']
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Modelo
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Avaliação
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Acurácia: {acc:.4f}")
print("Relatório:\n", classification_report(y_test, pred))

# Mostrar árvore
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=['Down', 'Up'], filled=True)
plt.show()
