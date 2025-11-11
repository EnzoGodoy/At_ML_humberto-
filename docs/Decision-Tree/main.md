# Modelo de Árvore de Decisão para Previsão de Movimentos do S\&P 500

## 1. Introdução

Este projeto busca desenvolver um modelo de **árvore de decisão** capaz de prever movimentos de valorização ou desvalorização no próximo pregão para cada uma das **500 ações que compõem o índice S\&P 500**. O objetivo é oferecer uma visão preditiva abrangente, aplicando um pipeline reproducível que cobre desde a exploração inicial até a avaliação e documentação dos resultados.

---

## 2. Tecnologias Utilizadas

* **Python 3.11**
* **pandas, numpy**
* **scikit-learn** (DecisionTreeClassifier, métricas)
* **matplotlib** (gráficos)
* **joblib** (salvar modelos)
* **kagglehub** ou entrada local de CSV

---

## 3. Metodologia e Etapas 

Abaixo cada etapa do projeto com exemplos práticos, entregáveis e métricas associadas (a tabela de avaliação original foi adaptada para o contexto das 500 ações).

### Etapa 1 — Exploração dos Dados 

* Objetivo: entender disponibilidade, qualidade e distribuição dos dados por ação.
* Entregáveis: tabela resumo por ticker (nº de observações, datas, missing), histogramas de retornos, série temporal do preço e heatmap de correlação para features.
* Exemplos de visualizações: `price over time`, `return distribution`, `missing value map`.

### Etapa 2 — Pré-processamento 

* Objetivo: tratar valores ausentes, padronizar colunas e alinhar datas entre tickers.
* Ações: forward/backward fill por ticker quando apropriado, remoção de linhas com valores críticos faltantes, normalização z-score para features usadas pelo modelo.

### Etapa 3 — Divisão dos Dados 

* Objetivo: separar treino/teste preservando ordem temporal para evitar data leakage.
* Abordagem: para cada ticker, dividir 80% inicial para treino e 20% final para teste. Alternativa: usar validação walk-forward (time-series cross-validation) para robustez.

### Etapa 4 — Treinamento do Modelo 

* Objetivo: treinar uma Decision Tree por ação ou um modelo global com ticker como feature.
* Estratégias:

  * **Modelo por ticker**: treina-se uma árvore para cada ação (mais preciso por ativo, mais custoso computacionalmente).
  * **Modelo global**: treina-se um único modelo usando dados concatenados e adicionando colunas auxiliares (`ticker_id`, `sector`), para capturar padrões cross-sectional.
* Hiperparâmetros: `max_depth`, `min_samples_leaf`, `criterion`.

### Etapa 5 — Avaliação do Modelo 

* Métricas por ticker e agregadas: **Acurácia**, **Precision**, **Recall**, **F1-score**, **Matriz de Confusão**, e **Sharpe ratio** de uma estratégia simples (opcional).
* Visualizações: matriz de confusão, curva de importâncias das features, distribuição das métricas pelos setores.


---

## 4. Limitações e Boas Práticas

* Risco de **overfitting** com árvores muito profundas; usar poda ou limitar `max_depth`.
* Perigo de **data leakage**: toda transformação que usa informação futura deve ser evitada ou aplicada somente no conjunto de treino.
* Modelos simples como Decision Trees têm capacidade limitada para capturar sinais fracos em mercados eficientes.

---

## 5. Código e Decision Tree
 Aqui você encontar o codigo da minha arvore e uma imagem dela 

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


---

## 6. Como executar 

1. Coloque um CSV consolidado `sandp500.csv` na raiz ou habilite `kagglehub` com credenciais.
2. Ajuste parâmetros no topo do script (`MAX_DEPTH`, `TEST_RATIO`, `LOCAL_CSV`).
3. Execute: `python pipeline_sp500_decision_tree.py`.

---

