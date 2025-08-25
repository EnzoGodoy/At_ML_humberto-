import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report


# Carregar o dataset diretamente do Kaggle
file_path = "constructor_results.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    file_path,
)

# Pré-processamento
def preprocess(df):
    df['points'].fillna(df['points'].median(), inplace=True)
    df['statusId'].fillna(df['statusId'].mode()[0], inplace=True)
    df['position'].fillna(df['position'].median(), inplace=True)
    features = ['constructorId', 'raceId', 'points', 'statusId', 'position']
    return df[features], df['position']

df = df.sample(n=100, random_state=42)
X, y = preprocess(df)

# Treinamento e avaliação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Exibir a árvore como texto
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)