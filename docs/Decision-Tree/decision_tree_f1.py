import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import kagglehub

plt.figure(figsize=(12, 10))

# Baixar e carregar o dataset do Kaggle
path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
df = pd.read_csv(path + "/constructor_results.csv")

# Selecionar features relevantes (removendo 'statusId')
x = df[['constructorId', 'raceId', 'points']]

# Codificar variáveis categóricas
label_encoder = LabelEncoder()
x['constructorId'] = label_encoder.fit_transform(x['constructorId'])
x['raceId'] = label_encoder.fit_transform(x['raceId'])

# Criar variável de saída categórica baseada nos pontos
y = pd.cut(
    df['points'],
    bins=[-1, 0, 10, df['points'].max()+1],  # faixas: sem pontos, poucos pontos, muitos pontos
    labels=['sem_pontos', 'poucos_pontos', 'muitos_pontos']
)

# Juntar features e target, remover linhas com NaN
data = x.copy()
data['target'] = y
data = data.dropna()

# Separar novamente features e target
x_clean = data.drop('target', axis=1)
y_clean = data['target']

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x_clean, y_clean, 
    test_size=0.2, 
    random_state=42
)

# Treinar árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier, feature_names=x_clean.columns, filled=True)

# Para imprimir na página HTML (SVG)
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())