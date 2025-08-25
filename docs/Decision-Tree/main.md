# Decision Tree com Dados da Fórmula 1

Este projeto utiliza o arquivo `constructor_results.csv` do Kaggle para treinar e avaliar uma árvore de decisão, prevendo a posição dos construtores em corridas de Fórmula 1.

## Como rodar

1. Instale as dependências:
   ```
   pip install pandas scikit-learn kaggle
   ```
2. Configure sua API Key do Kaggle em `C:\Users\SEU_USUARIO\.kaggle\kaggle.json`.
3. Execute o script:
   ```
   python decision_tree_f1.py
   ```

O script irá baixar automaticamente o arquivo do Kaggle (se necessário), realizar o pré-processamento dos dados e mostrar o relatório de classificação da árvore de decisão.

## Arquivos principais

- `decision_tree_f1.py`: Código principal do projeto.
- `main.md`: Esta documentação.

## Visualização da Árvore de Decisão

``` python exec="on" html="1"
--8<--"./docs/Decision-Tree/decision_tree_f1.py"
```

## Fonte

``` python exec="on" html="1"
--8<-- "./docs\Decision-Tree\decision_tree_f1.py"
```
