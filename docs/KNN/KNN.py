import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Caminho robusto (sempre pega o CSV da mesma pasta do script)
script_dir = os.path.dirname(__file__)  # pega a pasta do arquivo atual
file_path = os.path.join(script_dir, "AAPL_data.csv")

# 2. Carregar o CSV
df = pd.read_csv(file_path, parse_dates=['date'])

# 3. Ordenar por data
df = df.sort_values("date").reset_index(drop=True)

# 4. Criar coluna alvo: 1 se subir no próximo dia, 0 se cair
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Remover última linha (NaN no alvo porque não tem "próximo dia")
df = df.dropna(subset=["target"])

# 5. Features
features = ["open", "high", "low", "close", "volume"]
X = df[features]
y = df["target"]

# 6. Separar treino e teste (sem embaralhar tempo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 7. Criar e treinar o KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 8. Avaliar
y_pred = knn.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
