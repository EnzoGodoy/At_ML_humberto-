# Análise e Modelo de Árvores de Decisão para Previsão de Movimentos de Ações (S&P 500)

## 1. Introdução
O objetivo deste projeto foi construir um modelo de **árvore de decisão** para prever se o preço de fechamento de ações do índice S&P 500 subiria ou cairia no próximo dia. Utilizou-se o dataset público do Kaggle [S&P 500](https://www.kaggle.com/datasets/camnugent/sandp500), realizando limpeza, engenharia de features e modelagem com validação temporal.

---

## 2. Tecnologias Utilizadas
- **Python 3.11** para manipulação e modelagem  
- **Pandas** para pré-processamento  
- **scikit-learn** para construção da árvore de decisão  
- **Matplotlib** para visualizações  
- **KaggleHub** para aquisição do dataset  

---

## 3. Metodologia
O dataset foi baixado via `kagglehub` e passou por:
- Conversão de datas, remoção de valores ausentes e padronização de colunas  
- Criação de variáveis como retornos diários, médias móveis, volatilidade, RSI e variações intradiárias  
- Definição da variável alvo `target_up_next` (1 para alta, 0 para baixa)  
- Divisão temporal: 80% treino e 20% teste  
- Modelagem com `DecisionTreeClassifier` (`max_depth=6`, `random_state=42`)  

---

## 4. Resultados
- **Acurácia**: ~55–60%, variando por ação e janela temporal  
- **Importância de Features**: retornos recentes, volatilidade e RSI foram os indicadores mais relevantes  
- **Árvore de Decisão**:

```markdown
![Árvore de Decisão](docs/Decision-Tree/decision_tree.png)
```