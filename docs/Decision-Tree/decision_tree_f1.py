"""
Decision Tree simples para o dataset "S&P 500" (Kaggle: camnugent/sandp500)
Objetivo: limpar a base e treinar uma árvore de decisão para prever se o próximo dia terá alta.

Passos:
1) Baixa e carrega o dataset usando kagglehub.
2) Limpeza básica: remove NaN, converte datas, organiza colunas.
3) Criação de features básicas (retornos, médias móveis, volatilidade).
4) Split temporal treino/teste.
5) Treinamento da Decision Tree e avaliação.

Dependências: pandas, numpy, scikit-learn, matplotlib, kagglehub
"""

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Baixar dataset
print("Baixando dataset...")
DATASET_PATH = kagglehub.dataset_download("camnugent/sandp500")
csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado no dataset.")
file_path = os.path.join(DATASET_PATH, csv_files[0])
print(f"Usando arquivo: {file_path}")

# Carregar e limpar dados
df = pd.read_csv(file_path)

# Ajuste de colunas
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
if 'Name' in df.columns:
    symbol_col = 'Name'
else:
    symbol_col = None

# Ordenar por símbolo e data
if symbol_col:
    df = df.sort_values([symbol_col, 'date'])
else:
    df = df.sort_values('date')

# Preencher valores ausentes e remover linhas incompletas
df = df.dropna()

# Criar features básicas
df['Return_1d'] = df['close'].pct_change()
df['MA_5'] = df['close'].rolling(5).mean()
df['MA_10'] = df['close'].rolling(10).mean()
df['Volatility_10'] = df['close'].pct_change().rolling(10).std()

# Target: 1 se o próximo fechamento for maior que o atual
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Remover NaN restantes
df = df.dropna()

# Selecionar features
features = ['Return_1d', 'MA_5', 'MA_10', 'Volatility_10']
X = df[features]
y = df['Target']

# Split treino/teste (temporal)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Treinar árvore
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Avaliação
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Acurácia: {acc:.4f}")
print("Relatório:\n", classification_report(y_test, pred))

# Plotar árvore
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=features, class_names=['Down', 'Up'], filled=True)
plt.show()
