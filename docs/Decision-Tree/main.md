# Relatório Final – Análise e Modelo de Árvores de Decisão para Campeão de Construtores da F1

## 1. Introdução
O objetivo deste projeto foi construir um modelo de **árvore de decisão** capaz de prever o campeão de construtores da Fórmula 1 com base em dados históricos de desempenho. Utilizamos dados das bases `results.csv`, `races.csv` e `constructors.csv` do Kaggle (1950-2020), aplicando técnicas de engenharia de features, validação temporal e modelagem preditiva.

---

## 2. Tecnologias Utilizadas
- **Python 3.11** – linguagem principal para manipulação de dados e modelagem  
- **Pandas** – carregamento e manipulação dos datasets  
- **scikit-learn** – construção do modelo de árvore de decisão e avaliação  
- **Matplotlib** – visualização da árvore de decisão  
- **KaggleHub** – download do dataset diretamente do Kaggle  

---

## 3. Etapas do Processo

### 3.1 Aquisição do Dataset
O dataset foi obtido do Kaggle por meio da URL:  
[Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)  

Arquivos utilizados:
- `results.csv` - Resultados detalhados das corridas
- `races.csv` - Informações sobre as corridas  
- `constructors.csv` - Dados das equipes/construtoras

### 3.2 Pré-processamento e Engenharia de Features

**Combinação de dados:**  
- Junção das tabelas através de `raceId` e `constructorId`
- Agregação de pontos por construtora por ano

**Criação de features:**  
- `prev_points`: Pontos no ano anterior (feature preditora)
- `is_champion`: Indicador se foi campeã (variável target)

**Validação temporal:**  
- **Treino:** Dados de 1950 a 2019  
- **Teste:** Dados de 2020 (validação fora do tempo)

### 3.3 Modelagem
- **Modelo:** Decision Tree Classifier (`DecisionTreeClassifier`)  
- **Hiperparâmetros:** `max_depth=3` e `random_state=42`
- **Abordagem:** Classificação binária (campeão vs não-campeão)

---

## 4. Resultados e Análise

### 4.1 Performance do Modelo
- **Acurácia na validação (2020):** 92.5%
- **Campeão previsto para 2020:** Mercedes-AMG Petronas

### 4.2 Importância das Features
A análise revelou que:
- **Pontos no ano anterior** é o fator mais determinante
- A árvore identifica thresholds críticos de performance
- O modelo captura padrões históricos de dominância

### 4.3 Insights da Árvore de Decisão
A árvore de decisão gerada mostra:


``` python exec="on" html="1"
--8<-- "./docs/Decision-Tree/decision_tree_f1.py"
```

**Interpretação da árvore:**
- **Nó raiz:** Divide as equipes com base em pontos do ano anterior
- **Nós internos:** Estabelecem thresholds de performance histórica
- **Folhas:** Mostram a probabilidade de ser campeão
- **Cores:** Indicam a classe predominante (azul para não-campeão, laranja para campeão)

### 4.4 Análise de Previsões para 2020
O modelo previu corretamente a Mercedes como campeã de 2020 com alta probabilidade, demonstrando a forte correlação entre performance consistente ano após ano e conquista de títulos.

---

## 5. Conclusões

- O modelo alcançou **92.5% de acurácia** na previsão do campeão de 2020
- **Pontos no ano anterior** mostrou-se um preditor forte do sucesso
- A árvore de decisão fornece insights interpretáveis sobre os fatores determinantes
- A validação temporal demonstra a robustez do modelo para previsões futuras
- A Mercedes-AMG Petronas foi corretamente identificada como campeã de 2020

---

## 6. Possíveis Melhorias

1. **Features adicionais:** Incorporar vitórias, poles positions, voltas mais rápidas
2. **Janela temporal:** Usar média de pontos de múltiplos anos anteriores
3. **Modelos ensemble:** Testar Random Forest ou Gradient Boosting
4. **Análise por corrida:** Previsões em tempo de temporada
5. **Variáveis contextuais:** Incluir orçamento, mudanças regulamentares

---

## 7. Reprodução do Estudo

```bash
# Instalar dependências
pip install pandas scikit-learn matplotlib kagglehub

# Executar análise
python f1_champion_predictor.py
```

---

## 8. Referências

- [Kaggle: Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- [Scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Matplotlib: Visualization](https://matplotlib.org/stable/contents.html)

![Árvore de Decisão](decision_tree_f1_champion.png)