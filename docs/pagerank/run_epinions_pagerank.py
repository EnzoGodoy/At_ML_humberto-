
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
