

import opendatasets as od
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree


dataset_url = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"
od.download(dataset_url)

csv_path = "formula-1-world-championship-1950-2020/drivers.csv"


df = pd.read_csv(csv_path)


print(" Dimensão do dataset:", df.shape)
print("\n Tipos de variáveis:\n", df.dtypes)
print("\n Valores nulos por coluna:\n", df.isnull().sum())
print("\n Estatísticas gerais:\n", df.describe(include="all"))


num_cols = df.select_dtypes(include=["int64","float64"]).columns
df.hist(bins=20, figsize=(12,8))
plt.suptitle("Distribuição das Variáveis Numéricas")
plt.show()


df = df.loc[:, ~df.columns.duplicated()]

# Tratar nulos
for col in df.columns:
    if df[col].dtype in ["int64","float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


counts = df['nationality'].value_counts()
valid_nationalities = counts[counts >= 2].index
df_filtered = df[df['nationality'].isin(valid_nationalities)]


target_column = "nationality"
X = pd.get_dummies(df_filtered.drop(target_column, axis=1), drop_first=True)
y = df_filtered[target_column]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Treino X:", X_train.shape)
print("Teste X:", X_test.shape)
print("Treino y:", y_train.shape)
print("Teste y:", y_test.shape)


clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f" Acurácia do modelo: {accuracy:.2f}")

print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X_train.columns, class_names=clf.classes_, filled=True, rounded=True)
plt.show()
