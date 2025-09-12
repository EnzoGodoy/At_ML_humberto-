# Previsão de Tendência da Ação AAPL com KNN

Este projeto implementa um modelo de **KNN (K-Nearest Neighbors)** para prever a tendência (subida ou queda) da ação da **Apple (AAPL)** com base em dados históricos da bolsa de valores (preço e volume).

O desenvolvimento seguiu todas as etapas do processo de Machine Learning, desde a exploração dos dados até a avaliação do modelo, conforme os critérios fornecidos.

---

## Tecnologias Utilizadas
- **Python 3**
- **Pandas** → manipulação e análise de dados
- **Matplotlib / Seaborn** → visualizações gráficas
- **Scikit-learn** → implementação do modelo KNN, pré-processamento e métricas
- **GitHub Pages** → documentação e publicação do projeto

---

## Etapa 1 – Exploração dos Dados 

- Carregamento dos dados do arquivo `AAPL_data.csv`.  
- Verificação de valores ausentes.  
- Estatísticas descritivas (média, desvio padrão, mínimo, máximo).  
- Visualizações iniciais:
  - Evolução do **preço de fechamento ao longo do tempo**.
  - Histograma da distribuição do **volume negociado**.

``` python exec="on" html="1"
--8<-- "./docs/KNN/KNN_EDA.py"
```

---

## Etapa 2 – Pré-processamento 

Arquivo: [`KNN_model.py`](./KNN_model.py)

- Criação da variável **target**:
  - `1` se o preço **subir no dia seguinte**.
  - `0` se o preço **cair no dia seguinte**.
- Remoção de valores ausentes.
- **Normalização** das features (`StandardScaler`), pois o KNN é sensível à escala dos dados.

---

## Etapa 3 – Divisão dos Dados 

Arquivo: [`KNN_model.py`](./KNN_model.py)

- Divisão em **treino (80%)** e **teste (20%)**, mantendo a ordem temporal (sem shuffle).  
- Features utilizadas:
  - `open`, `high`, `low`, `close`, `volume`.

---

## Etapa 4 – Treinamento do Modelo 

Arquivo: [`KNN_model.py`](./KNN_model.py)

- Implementação do algoritmo **KNeighborsClassifier** com `n_neighbors=5`.  
- Treinamento feito sobre os dados normalizados de treino.

---

## Etapa 5 – Avaliação do Modelo 

Arquivo: [`KNN_model.py`](./KNN_model.py)

- Métricas utilizadas:
  - **Acurácia**.
  - **Relatório de classificação** (precisão, recall e F1-score).  
  - **Matriz de Confusão** (visualizada com Seaborn).  

---

## Visualização dos Resultados

Arquivo: [`KNN_plot.py`](./KNN_plot.py)

Foram criados gráficos adicionais para melhor interpretação dos resultados:

1. **Preço Real vs Previsões KNN**
   - Linha preta = preço real.
   - Setas verdes = previsão de **subida**.
   - Setas vermelhas = previsão de **queda**.

2. **Acertos e Erros do KNN**
   - Azul = previsão correta.
   - Vermelho = previsão incorreta.

Esses gráficos ajudam a entender visualmente onde o modelo acerta e erra na previsão da tendência da ação.

---


