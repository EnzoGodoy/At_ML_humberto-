# Decision Tree com Dados da Fórmula 1

Este projeto utiliza o arquivo `constructor_results.csv` do Kaggle para treinar e avaliar uma árvore de decisão, prevendo a posição dos construtores em corridas de Fórmula 1.

## 1. Exploração dos Dados

O arquivo `constructor_results.csv` possui os seguintes dados:

- **Total de registros:** 20.580
- **Colunas principais:**  
  - `raceId`: Identificador da corrida  
  - `constructorId`: Identificador do construtor  
  - `points`: Pontuação obtida  
  - `statusId`: Status do resultado  
  - `position`: Posição final (variável alvo)

### Estatísticas descritivas

| Coluna        | Média   | Mediana | Mínimo | Máximo |
|---------------|---------|---------|--------|--------|
| points        | 2.34    | 0.0     | 0      | 44     |
| position      | 7.2     | 7.0     | 1      | 22     |

A maioria dos construtores termina entre as primeiras posições, como mostra o histograma abaixo:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
df['position'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição das posições dos construtores')
plt.xlabel('Posição')
plt.ylabel('Contagem')
plt.show()
```

*Figura: Distribuição das posições dos construtores*

A análise inicial mostra que há predominância de posições baixas (1º, 2º, 3º), indicando forte competição entre os principais construtores.

---

## Como rodar

1. Instale as dependências:
   ```
   pip install pandas scikit-learn kaggle matplotlib kagglehub
   ```
2. Configure sua API Key do Kaggle em `C:\Users\SEU_USUARIO\.kaggle\kaggle.json`.
3. Execute o script:
   ```
   python decision_tree_f1.py
   ```

## Arquivos principais

- `decision_tree_f1.py`: Código principal do projeto.
- `main.md`: Esta documentação.

## Visualização da Árvore de Decisão



## Fonte

[Kaggle: Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=constructor_results.csv)


