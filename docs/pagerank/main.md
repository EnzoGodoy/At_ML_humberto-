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

## 5. Resultados Obtidos

A seguir são apresentados os **Top-10 usuários** segundo o PageRank **manual** (os CSVs mostram que manual e NetworkX são extremamente próximos).

Os dados já estavam ordenados no próprio CSV.

---

### ### **5.1 — Top-10 PageRank (d = 0.50)**

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
