import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import kagglehub

# Baixar dataset via KaggleHub
path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
print("Path to dataset files:", path)

# Carregar CSVs
results = pd.read_csv(os.path.join(path, "results.csv"))
races = pd.read_csv(os.path.join(path, "races.csv"))
constructors = pd.read_csv(os.path.join(path, "constructors.csv"))

# Juntar tabelas para ter ano + construtora + pontos
df = results.merge(races[['raceId', 'year']], on='raceId')
df = df.merge(constructors[['constructorId', 'name']], on='constructorId')

# Agregar pontos totais por construtora por ano
season_points = df.groupby(['year', 'constructorId', 'name'])['points'].sum().reset_index()

# Criar feature: pontos no ano anterior
season_points['prev_points'] = season_points.groupby('constructorId')['points'].shift(1)

# Criar target: campeão do ano
season_points['is_champion'] = season_points.groupby('year')['points'].transform(
    lambda x: x == x.max()
)

# Remover primeiras linhas sem histórico
season_points = season_points.dropna(subset=['prev_points'])

# Dividir em treino (até 2019) e teste (2020)
train = season_points[season_points['year'] < 2020]
test = season_points[season_points['year'] == 2020]

X_train = train[['prev_points']]
y_train = train['is_champion']
X_test = test[['prev_points']]
y_test = test['is_champion']

# Criar e treinar modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42, max_depth=3)  # max_depth controla complexidade
model.fit(X_train, y_train)

# Avaliar modelo em 2020
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia de validação (2020): {accuracy:.2f}")

# Mostrar ranking previsto
test['predicted'] = model.predict_proba(X_test)[:, 1]
print(test[['year', 'name', 'points', 'predicted']].sort_values('predicted', ascending=False))

# Visualizar Árvore
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['prev_points'], class_names=['Não Campeão', 'Campeão'], filled=True)
plt.show()
