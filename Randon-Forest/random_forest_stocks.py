import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# 1. Carregar dataset
# ==============================
# Construir caminho relativo ao arquivo atual para apontar ao CSV em ../KNN/
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.normpath(os.path.join(base_dir, os.pardir, "KNN", "all_stocks_5yr.csv"))
df = pd.read_csv(csv_path)
print(f"Usando arquivo CSV: {csv_path}")
print("Dimensões iniciais:", df.shape)
print(df.head())

# ==============================
# 2. Preparar dados
# ==============================
df = df.dropna(subset=["open", "high", "low", "close", "volume"])
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.iloc[:-1]

features = ["open", "high", "low", "close", "volume"]
X = df[features]
y = df["target"]

# ==============================
# 3. Dividir em treino e teste
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==============================
# 4. Treinar modelo
# ==============================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ==============================
# 5. Avaliar modelo
# ==============================
acc = accuracy_score(y_test, y_pred)
print(f"\n### RESULTADOS ###")
print(f"Acurácia: {acc:.3f}")
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 6. Matriz de confusão
# ==============================
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ==============================
# 7. Importância das variáveis
# ==============================
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=importances.index)
plt.title("Importância das Variáveis (Random Forest)")
plt.xlabel("Importância")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nImagens geradas: 'confusion_matrix.png' e 'feature_importance.png'")
print("\n✅ Execução concluída com sucesso!")
