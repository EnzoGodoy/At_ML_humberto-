import pandas as pd

# Carregar dados principais
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")
constructors = pd.read_csv("constructors.csv")

# Merge básico para ter ano + construtora
df = results.merge(races[['raceId', 'year']], on='raceId')
df = df.merge(constructors[['constructorId', 'name']], on='constructorId')

# Agregar pontos por construtora por temporada
season_points = df.groupby(['year', 'constructorId', 'name'])['points'].sum().reset_index()

# Ver os top 5 de 2019
print(season_points[season_points['year'] == 2019].sort_values('points', ascending=False).head())
