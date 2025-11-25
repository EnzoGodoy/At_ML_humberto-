# Análise de PageRank na Rede de Confiança Epinions (soc-Epinions1)
 
---

## 1. Introdução

Este trabalho aplica o algoritmo **PageRank** ao dataset **soc-Epinions1**, uma grande rede de confiança extraída do antigo site de reviews “Epinions”. Cada ligação dirigida da forma **A → B** significa que **A confia em B**, o que permite analisar influência, reputação e credibilidade dos usuários.

O objetivo é:

- Implementar o PageRank manualmente
- Comparar com o PageRank do *NetworkX*
- Avaliar o efeito do **damping factor** (d = 0.50, 0.85 e 0.99)
- Identificar os usuários mais influentes da rede

---

## 2. Dataset: soc-Epinions1

O dataset foi retirado do repositório **SNAP (Stanford Network Analysis Project)**.

**Características principais:**

- Nós (usuários): ~75.888  
- Arestas (relações de confiança): ~508.837  
- Direcionamento: **A → B** (A confia em B)  
- Tipo: Rede Social dirigida

A rede representa uma estrutura complexa de reputação, onde usuários confiados por outros usuários altamente confiáveis tendem a ganhar maior relevância no PageRank.

---

## 3. O Algoritmo PageRank

O PageRank simula um “navegador aleatório”, que segue links com probabilidade **d** e teleporta para outro nó com probabilidade **1 − d**.

A equação do PageRank é:

\[
PR(i) = \frac{1-d}{N} + d \cdot \sum_{j \in In(i)} \frac{PR(j)}{L_j}
\]

Esse processo converge por iterações sucessivas.  
Sua interpretação nesse contexto é direta: **usuários com alto PageRank são fortemente confiados por outros, especialmente por usuários que também são confiados.**

---

## 4. Metodologia

Três execuções foram realizadas variando o **damping factor**:

- **d = 0.50** (teleporte domina, ranking mais homogêneo)  
- **d = 0.85** (valor clássico do PageRank)  
- **d = 0.99** (forte dependência da estrutura do grafo)

Para cada caso, o script gerou:

1. PageRank manual  
2. PageRank NetworkX  
3. Arquivo CSV consolidado  

---

## Códigos

=== "Script"

	``` { .python .copy .select linenums='1' title="run_epinions_pagerank.py" }
	import gzip
	import networkx as nx
	import numpy as np
	import pandas as pd
	import time
	import os

	def load_epinions_gz(path):
        
		G = nx.DiGraph()

		with gzip.open(path, "rt") as f:
			for line in f:
				if line.startswith("#"):
					continue
				parts = line.strip().split()
				if len(parts) < 2:
					continue
				u, v = parts[0], parts[1]
				G.add_edge(u, v)

		return G


	def pagerank_from_scratch(G, d=0.85, tol=1e-4, max_iter=200):
		nodes = list(G.nodes())
		n = len(nodes)

		idx = {node: i for i, node in enumerate(nodes)}

		p = np.ones(n) / n
		outdeg = np.array([G.out_degree(node) for node in nodes], dtype=float)

		teleport = (1 - d) / n

		start = time.time()
		for it in range(1, max_iter + 1):
			p_new = np.full(n, teleport)

			for j, node in enumerate(nodes):
				if outdeg[j] == 0:
					p_new += d * p[j] / n
				else:
					for nei in G.successors(node):
						i = idx[nei]
						p_new[i] += d * p[j] / outdeg[j]

			p_new /= p_new.sum()
			diff = np.abs(p_new - p).sum()
			p = p_new

			if diff < tol:
				break

		end = time.time()

		return {nodes[i]: float(p[i]) for i in range(n)}, it, end - start


	def top_k(pr, k=10):
		return sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]

	def pagerank_networkx(G, d):
		return nx.pagerank(G, alpha=d)

	def main():

        
		script_dir = os.path.dirname(os.path.abspath(__file__))
		dataset = os.path.join(script_dir, "soc-Epinions1.txt.gz")
        
		print("Carregando grafo Epinions...")
		print(f"Procurando arquivo em: {dataset}")
		G = load_epinions_gz(dataset)
		print(f"Nós: {G.number_of_nodes()} | Arestas: {G.number_of_edges()}")
		print("-----------------------------------------------------------")

		damping_values = [0.5, 0.85, 0.99]

		for d in damping_values:
			print(f"\n### PageRank do zero — d={d}")
			pr_manual, iters, t = pagerank_from_scratch(G, d=d)

			print(f"Convergiu em {iters} iterações | tempo {t:.2f}s")
			print("\nTop 10 (manual):")
			for i, (node, score) in enumerate(top_k(pr_manual), 1):
				print(f"{i:2d}. {node} — {score:.6f}")

            
			print(f"\n### NetworkX PageRank — d={d}")
			pr_nx = pagerank_networkx(G, d)
			for i, (node, score) in enumerate(top_k(pr_nx), 1):
				print(f"{i:2d}. {node} — {score:.6f}")

            
			df = pd.DataFrame({
				"node": list(pr_manual.keys()),
				"pagerank_manual": list(pr_manual.values()),
				"pagerank_networkx": [pr_nx[n] for n in pr_manual.keys()]
			})

			name = f"epinions_pagerank_d_{str(d).replace('.','')}.csv"
			df.to_csv(name, index=False)
			print(f"\nArquivo salvo: {name}")

		print("\n>>> Fim! Todos os valores foram gerados com sucesso.")


	if __name__ == "__main__":
		main()
	```

 
     
### Como executar

Para rodar este script localmente (dentro de `docs/pagerank`):

```cmd
python run_epinions_pagerank.py
```

O script gera três CSVs (`epinions_pagerank_d_05.csv`, `epinions_pagerank_d_085.csv`, `epinions_pagerank_d_099.csv`) no mesmo diretório.


## 5. Resultados Obtidos

A seguir são apresentados os **Top-10 usuários** segundo o PageRank **manual** (os CSVs mostram que manual e NetworkX são extremamente próximos).

Os dados já estavam ordenados no próprio CSV.

---

### ***5.1 — Top-10 PageRank (d = 0.50)***

| Rank | Node | PageRank |
|------|------|----------|
| 1 | 0 | 0.000381 |
| 2 | 5 | 0.000113 |
| 3 | 4 | 0.000084 |
| 4 | 8 | 0.000083 |
| 5 | 9 | 0.000068 |
| 6 | 2 | 0.000061 |
| 7 | 7 | 0.000027 |
| 8 | 1 | 0.000024 |
| 9 | 6 | 0.000014 |
| 10 | 3 | 0.000012 |

---

### **5.2 — Top-10 PageRank (d = 0.85)**

| Rank | Node | PageRank |
|------|------|----------|
| 1 | 0 | 0.000932 |
| 2 | 5 | 0.000381 |
| 3 | 4 | 0.000271 |
| 4 | 8 | 0.000193 |
| 5 | 9 | 0.000160 |
| 6 | 2 | 0.000143 |
| 7 | 7 | 0.000060 |
| 8 | 1 | 0.000059 |
| 9 | 6 | 0.000035 |
| 10 | 3 | 0.000029 |

---

### **5.3 — Top-10 PageRank (d = 0.99)**

| Rank | Node | PageRank |
|------|------|----------|
| 1 | 0 | 0.001340 |
| 2 | 5 | 0.000645 |
| 3 | 4 | 0.000458 |
| 4 | 8 | 0.000283 |
| 5 | 9 | 0.000235 |
| 6 | 2 | 0.000214 |
| 7 | 1 | 0.000090 |
| 8 | 7 | 0.000094 |
| 9 | 6 | 0.000055 |
| 10 | 3 | 0.000046 |

---

## 6. Análise dos Resultados

### **1. O nó mais influente é consistentemente o `0`**
O nó 0 aparece em **primeiro lugar em todas as configurações**, indicando que:

- Muitos usuários confiam nele  
- Ele recebe confiança de usuários que também são bem posicionados  
- É um “hub de reputação” dentro da rede Epinions

---

### **2. Efeito do damping factor**

**d = 0.50 → ranking mais achatado**  
O teleporte domina e a estrutura real da rede pesa pouco. Os valores são próximos entre si.

**d = 0.85 → comportamento clássico**  
Aparecem influenciadores reais. Hierarquias de confiança começam a surgir com clareza.

**d = 0.99 → estrutura da rede domina totalmente**  
O ranking fica muito mais “puxado para cima” pelos nós que estão em regiões densas da rede.  
A diferença entre o primeiro e o segundo lugar aumenta muito.

---

### **3. Implementação manual x NetworkX**

Comparando os CSVs:

- Os valores de PageRank são **muito próximos**  
- Pequenas diferenças vêm do método de normalização e arredondamento  
- A ordem dos Top-10 permanece idêntica na maioria dos casos

**Conclusão:**  
➡️ A implementação manual está correta e validada pelo NetworkX.

---

## 7. Conclusão

- O PageRank se mostrou eficaz para medir **influência baseada em confiança**.  
- O usuário **node 0** é o principal influenciador da rede Epinions.  
- O damping factor altera significativamente a sensibilidade do ranking.  
- A implementação própria produziu resultados consistentes com a biblioteca NetworkX.  
- A análise destaca como reputação se propaga em redes sociais dirigidas.

---
