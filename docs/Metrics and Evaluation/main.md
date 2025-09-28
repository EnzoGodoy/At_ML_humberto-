# K-Means Evaluation — main.md

## 1. Resumo do projeto

Este projeto avalia a aplicação do algoritmo **K‑Means** ao dataset `all_stocks_5yr.csv` (agrupamento de ações por retorno médio e volatilidade). Foi criado um script (`kmeans_evaluation.py`) que executa K‑Means para vários valores de K, calcula métricas internas de cluster e gera gráficos e arquivos CSV com os resultados.

## 2. Objetivo

* Aplicar K‑Means para agrupar empresas/ações segundo comportamento (retorno médio × volatilidade).
* Avaliar a qualidade dos clusters usando métricas internas (não supervisionadas) e visualizações.
* Gerar artefatos para incluir no relatório final: tabelas, gráficos e interpretações.

## 3. Métricas utilizadas e justificativa

* **Inertia / WCSS (Within‑cluster Sum of Squares)**: usado no *Elbow Method* para sugerir valores de K; quantifica variabilidade interna dos clusters.
* **Silhouette score (média e por amostra)**: mede coesão e separação; valores próximos de 1 indicam clusters bem definidos, valores próximos a 0 indicam sobreposição, negativos indicam má atribuição.
* **Calinski‑Harabasz Index**: razão de variância entre e dentro dos clusters; quanto maior, melhor.
* **Davies‑Bouldin Index**: média da similaridade entre cada cluster e seu mais similar; quanto menor, melhor.
* **Tamanho dos clusters**: detectar clusters muito pequenos (possíveis outliers) ou muito grandes (subgrupos ocultos).
* **Centroides**: interpretar características centrais de cada cluster (ex.: cluster com baixa volatilidade e alto retorno).

## 4. Arquivos / Saídas geradas pelo script

Ao rodar `kmeans_evaluation.py`, será criada uma pasta `kmeans_evaluation_outputs/` contendo:

* `kmeans_metrics_by_k.csv` — métricas (inertia, silhouette_mean, calinski_harabasz, davies_bouldin) para cada K testado.
* `elbow_silhouette_by_k.png` — gráfico com Elbow (inertia) e Silhouette média por K.
* `kmeans_summary.csv` — resumo das métricas para o K escolhido (por padrão K=3).
* `clusters_result.csv` — lista completa de tickers e o cluster atribuído.
* `cluster_summary.csv` — resumo por cluster (contagem, média de volatilidade, média de retorno).
* `clusters_scatter_with_centroids.png` — scatter plot (volatilidade × retorno) com clusters e centroides.
* `silhouette_plot.png` — silhouette plot com distribuição por cluster.

### Visualizações geradas (inseridas automaticamente pelo script)


![Elbow e Silhouette](kmeans_evaluation_outputs/elbow_silhouette_by_k.png)
*Elbow (Inertia) e Silhouette média por K — ajuda a selecionar K.*

![Scatter com centroides](kmeans_evaluation_outputs/clusters_scatter_with_centroids.png)
*Scatter plot (volatilidade × retorno) com os clusters encontrados e centroides marcados.*

![Silhouette plot](kmeans_evaluation_outputs/silhouette_plot.png)
*Silhouette por amostra, mostrando a qualidade das atribuições por cluster.*

## 5. Como rodar (passo a passo)

1. Coloque `kmeans_evaluation.py` no diretório do repositório (ou ajuste o caminho `csv_path` dentro do script para apontar ao CSV correto).
2. Abra um terminal na pasta do script.
3. Rode: `python kmeans_evaluation.py` (ou `python3` conforme seu ambiente).
4. Verifique a pasta `kmeans_evaluation_outputs/` para os CSVs e imagens.
5. Commit/push da pasta `kmeans_evaluation_outputs/` (ou copie as imagens para uma pasta do repositório) para que as imagens apareçam diretamente no GitHub ao visualizar `main.md`.

## 6. Interpretação rápida dos resultados (o que incluir no relatório)

* Compare *Inertia* e *Silhouette mean* para escolher K: o *elbow* sugere K onde a redução de inertia diminui abruptamente; o K que maximiza a silhouette média também é um bom candidato.
* Use o *silhouette per sample* para identificar ações mal atribuídas (valores negativos ou muito baixos) e discuta possíveis razões (outliers, features insuficientes).
* Quando Calinski e Davies‑Bouldin concordarem com silhouette, a confiança no K escolhido aumenta. Caso haja divergência, discuta trade‑offs.
* Analise os centroides e tamanhos de cluster para fornecer interpretações qualitativas (ex.: Cluster A = baixa volatilidade, retorno moderado → perfil conservador).

## 7. Sugestões de melhorias e próximos passos

* **Padronização**: testar `StandardScaler` nas features (mean_return, volatility) e comparar métricas com/sem escalonamento.
* **Adicionar features**: incluir volume, indicadores técnicos ou fatores do mercado para enriquecer o espaço de características.
* **Estabilidade**: rodar KMeans várias vezes com seeds diferentes e avaliar consistência dos rótulos (stability analysis).
* **Validação externa**: caso existam rótulos externos (setor, segmento), calcular métricas externas (ARI, AMI) para complementar a avaliação.

## 8. Trecho de código (exemplo de execução rápida)

```python
# executar avaliação básica (exemplo simplificado)
from kmeans_evaluation import main  # se transformar o script em módulo
# ou simplesmente
# python kmeans_evaluation.py
```

## 9. Referências

* Repositório analisado: `all_stocks_5yr.csv` (implementação original do exercício K‑Means).
* Material de métricas e conceitos (para apoiar o relatório): artigos e documentação sobre Silhouette, Calinski‑Harabasz, Davies‑Bouldin e Elbow Method.

---

> Se quiser, eu posso também gerar o arquivo `main.md` pronto para download (arquivo .md) ou ajustar legendas/captions das imagens. Quer que eu gere o `.md` para você baixar?
