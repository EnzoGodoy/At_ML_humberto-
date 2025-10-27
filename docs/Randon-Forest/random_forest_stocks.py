import pandas as pd
import numpy as np
import random
from collections import Counter
import time


CSV_PATH = "c:/Users/enzo.godoy/Documents/GitHub/At_ML_humberto-/docs/KNN/all_stocks_5yr.csv"


# Parâmetros da Random Forest
N_TREES = 2            # Reduzido para teste rápido (aumente para 10+ em produção)
MAX_DEPTH = 3          # Profundidade reduzida para maior velocidade
MIN_SIZE = 20          # Aumentado para splits mais rápidos
FEATURE_SUBSAMPLE = 2  # Usar apenas 2 features por split para teste rápido

# Configuração de execução rápida
FAST_RUN = True        # Se True, usa apenas uma fração dos dados
SAMPLE_FRACTION = 0.1  # Usar apenas 10% dos dados para teste rápido
TEST_SIZE = 0.25       
RANDOM_SEED = 42

APPLY_MINMAX = True    

# =========================
# Utilitários e funções
# =========================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Funções de carregamento e preparação de dados
def load_and_prepare(csv_path):
    """Carrega dados do CSV e prepara features"""
    print("Carregando dados...")
    df = pd.read_csv(csv_path)

    required_cols = {"open", "high", "low", "close", "volume", "Name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV deve ter colunas: {required_cols}. Achado: {df.columns.tolist()}")

    # Target: 1 se preço fechou maior que abertura
    df["target"] = (df["close"] > df["open"]).astype(int)

    # Ordenar por data e calcular prev_close por ação
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["Name", "date"]).reset_index(drop=True)
    df["prev_close"] = df.groupby("Name")["close"].shift(1)

    features = ["open", "high", "low", "volume", "prev_close"]
    df = df.dropna(subset=features + ["target"]).reset_index(drop=True)

    return df, features

def minmax_scale(df, feature_cols):
    """Normaliza features para [0,1]"""
    print("Normalizando features...")
    for col in feature_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0
    return df

def train_test_split_df(df, test_size, seed=42):
    """Divide em treino/teste preservando ordem temporal"""
    print("Separando em treino/teste...")
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = int(len(df_shuffled) * test_size)
    test = df_shuffled.iloc[:n_test].reset_index(drop=True)
    train = df_shuffled.iloc[n_test:].reset_index(drop=True)
    return train, test


# =========================
# Implementação do Random Forest
# =========================

# Helper: Calculate Gini impurity
def gini_impurity(y):
    if not y:
        return 0
    counts = Counter(y)
    impurity = 1
    for count in counts.values():
        prob = count / len(y)
        impurity -= prob ** 2
    return impurity

# Helper: Split dataset based on feature index and value
def split_dataset(X, y, feature_idx, value):
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature_idx] <= value:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return left_X, left_y, right_X, right_y

# Decision Tree Node
class Node:
    def __init__(self, feature_idx=None, value=None, left=None, right=None, label=None):
        self.feature_idx = feature_idx
        self.value = value
        self.left = left
        self.right = right
        self.label = label  # Leaf node label

# Build a single decision tree (recursive)
def build_tree(X, y, max_depth, min_samples_split, max_features):
    if len(y) < min_samples_split or max_depth == 0:
        return Node(label=Counter(y).most_common(1)[0][0])

    n_features = len(X[0])
    features = random.sample(range(n_features), max_features)  # Random subset

    best_gini = float('inf')
    best_feature_idx, best_value = None, None
    best_left_X, best_left_y, best_right_X, best_right_y = None, None, None, None

    for feature_idx in features:
        values = sorted(set(row[feature_idx] for row in X))
        for value in values:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature_idx, value)
            if not left_y or not right_y:
                continue
            p_left = len(left_y) / len(y)
            gini = p_left * gini_impurity(left_y) + (1 - p_left) * gini_impurity(right_y)
            if gini < best_gini:
                best_gini = gini
                best_feature_idx = feature_idx
                best_value = value
                best_left_X, best_left_y = left_X, left_y
                best_right_X, best_right_y = right_X, right_y

    if best_gini == float('inf'):
        return Node(label=Counter(y).most_common(1)[0][0])

    left = build_tree(best_left_X, best_left_y, max_depth - 1, min_samples_split, max_features)
    right = build_tree(best_right_X, best_right_y, max_depth - 1, min_samples_split, max_features)
    return Node(best_feature_idx, best_value, left, right)

# Predict with a single tree
def predict_tree(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature_idx] <= node.value:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

# Random Forest class
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        print(f"Treinando {self.n_estimators} árvores...")
        n_samples = len(X)
        n_features = len(X[0])
        max_features = int(n_features ** 0.5) if self.max_features == 'sqrt' else self.max_features

        for i in range(self.n_estimators):
            print(f"Árvore {i+1}/{self.n_estimators}...")
            start = time.time()
            # Bootstrap sample
            bootstrap_idx = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_boot = [X[i] for i in bootstrap_idx]
            y_boot = [y[i] for i in bootstrap_idx]
            tree = build_tree(X_boot, y_boot, self.max_depth, self.min_samples_split, max_features)
            self.trees.append(tree)
            print(f"  → Treinada em {time.time() - start:.2f}s")

    def predict(self, X):
        predictions = []
        for x in X:
            tree_preds = [predict_tree(tree, x) for tree in self.trees]
            predictions.append(Counter(tree_preds).most_common(1)[0][0])
        return predictions


# Métricas de avaliação
def accuracy_metric(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / float(len(actual)) if actual else 0.0

def confusion_matrix(actual, predicted):
    cm = Counter()
    for a, p in zip(actual, predicted):
        cm[(a, p)] += 1
    return dict(cm)

if __name__ == "__main__":
    # Carregar e preparar os dados
    print("Carregando dados...")
    df, features = load_and_prepare(CSV_PATH)
    
    # Aplicar normalização se configurado
    if APPLY_MINMAX:
        print("Aplicando normalização Min-Max...")
        df = minmax_scale(df, features)
    
    # Separar em treino e teste
    print("Separando dados em treino e teste...")
    train_df, test_df = train_test_split_df(df, TEST_SIZE, RANDOM_SEED)
    
    # Preparar dados para o formato esperado pelo algoritmo
    train_data = train_df[features + ["target"]].values.tolist()
    test_data = test_df[features + ["target"]].values.tolist()
    
    # Configurar número de features para cada split se não especificado
    if FEATURE_SUBSAMPLE is None:
        n_features = int(np.sqrt(len(features)))
    else:
        n_features = FEATURE_SUBSAMPLE
    
    # Informações e amostragem para execução rápida (debug)
    print(f"Tamanho treino: {len(train_data)} linhas | Tamanho teste: {len(test_data)} linhas")
    if FAST_RUN:
        n_sample = max(1, int(len(train_data) * SAMPLE_FRACTION))
        print(f"FAST_RUN ativo: usando {n_sample} linhas de treino (fração={SAMPLE_FRACTION}) para debug)")
        train_data_small = random.sample(train_data, n_sample)
    else:
        train_data_small = train_data

    # Treinar a Random Forest (versão simplificada)
    print(f"Treinando Random Forest com {N_TREES} árvores...")
    rf = RandomForest(n_estimators=N_TREES,
                      max_depth=MAX_DEPTH,
                      min_samples_split=MIN_SIZE,
                      max_features='sqrt' if FEATURE_SUBSAMPLE is None else FEATURE_SUBSAMPLE)
    t0 = time.time()
    X_train = [row[:-1] for row in train_data_small]
    y_train = [row[-1] for row in train_data_small]
    rf.fit(X_train, y_train)
    t_total = time.time() - t0
    print(f"Treinamento completo em {t_total:.2f}s")

    # Fazer predições
    print("Fazendo predições...")
    X_test = [row[:-1] for row in test_data]
    predictions = rf.predict(X_test)
    actual = [row[-1] for row in test_data]

    # Calcular métricas
    accuracy = accuracy_metric(actual, predictions)
    conf_matrix = confusion_matrix(actual, predictions)

    # Imprimir resultados
    print("\nResultados:")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nMatriz de Confusão:")
    print("Real \\ Predito |  0  |  1  |")
    print("-" * 30)
    print(f"      0       | {conf_matrix.get((0,0), 0):3d} | {conf_matrix.get((0,1), 0):3d} |")
    print(f"      1       | {conf_matrix.get((1,0), 0):3d} | {conf_matrix.get((1,1), 0):3d} |")
