import opendatasets as od
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import numpy as np

# Download do dataset
dataset_url = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"
od.download(dataset_url)

# Carregar os dados relevantes
constructors_df = pd.read_csv("formula-1-world-championship-1950-2020/constructors.csv")
constructor_standings_df = pd.read_csv("formula-1-world-championship-1950-2020/constructor_standings.csv")
races_df = pd.read_csv("formula-1-world-championship-1950-2020/races.csv")
results_df = pd.read_csv("formula-1-world-championship-1950-2020/results.csv")

# Preparar os dados para análise
# Juntar as tabelas para obter informações completas
merged_df = constructor_standings_df.merge(
    races_df[['raceId', 'year', 'name', 'round']], 
    on='raceId'
).merge(
    constructors_df[['constructorId', 'name', 'nationality']], 
    on='constructorId', 
    suffixes=('_race', '_constructor')
)

# Filtrar para anos recentes (a partir de 2000)
recent_years_df = merged_df[merged_df['year'] >= 2000].copy()

# Calcular estatísticas por equipe por ano
team_stats = recent_years_df.groupby(['year', 'constructorId', 'name_constructor']).agg({
    'points': 'max',  # Pontuação máxima no campeonato
    'wins': 'max',    # Máximo de vitórias
    'position': 'min' # Melhor posição no campeonato
}).reset_index()

# Adicionar uma coluna indicando se a equipe foi campeã (position = 1)
team_stats['champion'] = (team_stats['position'] == 1).astype(int)

# Preparar dados para treinamento
# Criar features baseadas no desempenho histórico
historical_stats = team_stats.groupby('constructorId').agg({
    'points': ['mean', 'max', 'min'],
    'wins': ['mean', 'max'],
    'position': ['mean', 'min'],
    'champion': 'sum'
}).reset_index()

historical_stats.columns = ['constructorId', 'avg_points', 'max_points', 'min_points', 
                           'avg_wins', 'max_wins', 'avg_position', 'best_position', 'championships']

# Juntar com informações das equipes
constructor_info = constructors_df[['constructorId', 'name', 'nationality']]
historical_stats = historical_stats.merge(constructor_info, on='constructorId')

# Preparar dados para 2021 (último ano completo antes da previsão)
latest_data = team_stats[team_stats['year'] == 2021].merge(
    historical_stats, on='constructorId', suffixes=('_2021', '_historical')
)

# Features para o modelo
feature_columns = ['points_2021', 'wins_2021', 'position_2021', 
                   'avg_points', 'max_points', 'min_points', 
                   'avg_wins', 'max_wins', 'avg_position', 'best_position', 'championships']

X = latest_data[feature_columns]
y = latest_data['champion_2021']  # Se foi campeão em 2021

# Dividir os dados (usando todos os dados para treino já que temos poucos)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinar o modelo
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Visualizar a árvore de decisão
plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf, 
    feature_names=feature_columns, 
    class_names=['Não Campeão', 'Campeão'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão para Prever o Campeão de Construtores da F1")
plt.show()

# Mostrar importância das features
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("Importância das features:")
print(feature_importance)

# Prever o campeão de 2021
champion_2021 = latest_data[latest_data['champion_2021'] == 1]['name_constructor'].values[0]
print(f"\nO campeão de construtores de 2021 foi: {champion_2021}")

# Análise dos dados
print("\n--- Análise dos Dados ---")
print(f"Total de equipes na F1 (2000-2021): {latest_data.shape[0]}")
print(f"Equipes que já foram campeãs: {historical_stats['championships'].sum()}")

# Gráfico de distribuição de pontos
plt.figure(figsize=(12, 6))
sns.boxplot(data=latest_data, x='champion_2021', y='points_2021')
plt.title('Distribuição de Pontos por Status de Campeão (2021)')
plt.xticks([0, 1], ['Não Campeão', 'Campeão'])
plt.ylabel('Pontos')
plt.xlabel('')
plt.show()