import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Carregar os dados
df = pd.read_csv('constructor_results.csv')

# Pré-processamento
def preprocess(df):
    # Preencher valores ausentes
    df['points'].fillna(df['points'].median(), inplace=True)
    df['statusId'].fillna(df['statusId'].mode()[0], inplace=True)
    df['position'].fillna(df['position'].median(), inplace=True)

    # Selecionar features
    features = ['constructorId', 'raceId', 'points', 'statusId', 'position']
    return df[features], df['position']

# Selecionar amostra pequena para exemplo
df = df.sample(n=100, random_state=42)

X, y = preprocess(df)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar árvore de decisão
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Avaliar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))