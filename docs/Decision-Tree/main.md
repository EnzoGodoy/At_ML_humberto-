# Modelo de Árvore de Decisão para Previsão de Movimentos do S&P 500

## 1. Introdução
Este projeto tem como objetivo desenvolver um modelo de **árvore de decisão** para prever se o preço de fechamento diário das ações do índice **S&P 500** apresentará valorização ou desvalorização no próximo pregão.  
Foi utilizado o dataset público do [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500), passando por etapas de **limpeza de dados**, **engenharia de features** e **modelagem preditiva** com validação temporal.

---

## 2. Tecnologias Utilizadas
- **Python 3.11**: linguagem principal para análise e modelagem  
- **pandas**: tratamento e pré-processamento de dados  
- **numpy**: operações matemáticas e vetorização  
- **scikit-learn**: implementação do `DecisionTreeClassifier`  
- **matplotlib**: geração de visualizações gráficas  
- **kagglehub**: aquisição programática do dataset

---

## 3. Metodologia
1. **Aquisição e Limpeza de Dados**: conversão de datas, remoção de valores ausentes e padronização de colunas.  
2. **Engenharia de Features**: criação de indicadores como retornos diários, médias móveis, volatilidade, RSI e variações intradiárias.  
3. **Definição da Variável-Alvo**: `target_up_next` (1 para valorização, 0 para desvalorização).  
4. **Divisão Temporal dos Dados**: 80% para treino e 20% para teste, preservando a ordem cronológica para evitar **data leakage**.  
5. **Modelagem**: uso de `DecisionTreeClassifier` com `max_depth=6` e `random_state=42`.

---

## 4. Resultados
- **Acurácia**: aproximadamente 55–60%, variando conforme a ação e a janela temporal analisada.  
- **Importância das Features**: retornos recentes, volatilidade e RSI se destacaram como indicadores relevantes.  
- **Visualização da Árvore de Decisão**:  

![Decision Tree](decision_tree.png)

```python 

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

```