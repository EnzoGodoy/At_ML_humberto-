import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === 1. Carregar o CSV ===
df = pd.read_csv("all_stocks_5yr.csv")

# === 2. Limpeza e preparação dos dados ===
# Ordenar por empresa e data
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['Name', 'date'])

# Criar coluna de fechamento do próximo dia
df['close_next'] = df.groupby('Name')['close'].shift(-1)

# Criar variável alvo: 1 se subir, 0 se cair
df['target'] = (df['close_next'] > df['close']).astype(int)

# Remover linhas com valores ausentes
df = df.dropna()

# === 3. Selecionar features ===
features = ['open', 'high', 'low', 'close', 'volume']
X = df[features]
y = df['target']

# === 4. Dividir em treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Treinar o modelo ===
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# === 6. Avaliar ===
predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Acurácia: {acc:.4f}")

# === 7. Importância das features ===
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# === 8. Plotar gráfico das importâncias ===
plt.figure(figsize=(8, 5))
importances.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Importância das Features - Random Forest')
plt.ylabel('Importância Relativa')
plt.xlabel('Variáveis')
plt.tight_layout()

# Salvar imagem para uso no main
plt.savefig("feature_importances.png", dpi=300)
plt.show()
