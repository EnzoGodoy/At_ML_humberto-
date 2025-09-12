import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Caminho robusto
script_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
file_path = os.path.join(script_dir, "AAPL_data.csv")

df = pd.read_csv(file_path, parse_dates=['date'])
df = df.sort_values("date").reset_index(drop=True)

# Criar target
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna(subset=["target"])

# Features
features = ["open", "high", "low", "close", "volume"]
X = df[features]
y = df["target"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# Treinar
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Previsões
df_test = df.iloc[len(X_train):].copy()
df_test["prediction"] = knn.predict(X_test)
df_test["correct"] = df_test["prediction"] == df_test["target"]

# Plot Preço + Previsões
plt.figure(figsize=(12,6))
plt.plot(df_test["date"], df_test["close"], label="Preço Real", color="black")
plt.scatter(df_test["date"][df_test["prediction"]==1],
            df_test["close"][df_test["prediction"]==1],
            label="Previsto Subida", color="green", marker="^", alpha=0.7)
plt.scatter(df_test["date"][df_test["prediction"]==0],
            df_test["close"][df_test["prediction"]==0],
            label="Previsto Queda", color="red", marker="v", alpha=0.7)
plt.title("Previsões KNN sobre AAPL")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Acertos x Erros
plt.figure(figsize=(12,5))
plt.scatter(df_test["date"], df_test["close"],
            c=df_test["correct"], cmap="bwr", alpha=0.6)
plt.title("Acertos (azul) e Erros (vermelho) - KNN")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento (USD)")
plt.show()
