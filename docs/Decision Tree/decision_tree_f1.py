import pandas as pd
import os

# Baixar o arquivo do Kaggle (executa só se não existir)
if not os.path.exists('constructor_results.csv'):
    os.system('kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 --unzip')
    # O arquivo estará na pasta após unzip

# Carregar os dados
df = pd.read_csv('constructor_results.csv')

# Pré-processamento
def preprocess(df):
    df['points'].fillna(df['points'].median(), inplace=True)
    df['statusId'].fillna(df['statusId'].mode()[0], inplace=True)
    df['position'].fillna(df['position'].median(), inplace=True)
    features = ['constructorId', 'raceId', 'points', 'statusId', 'position']
    return df[features], df['position']

df = df.sample(n=100, random_state=42)
X, y = preprocess(df)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))