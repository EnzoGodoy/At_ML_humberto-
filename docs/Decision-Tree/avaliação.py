

import opendatasets as od
import pandas as pd
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from io import StringIO


dataset_url = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"
od.download(dataset_url)
csv_path = "formula-1-world-championship-1950-2020/constructor_standings.csv"


df = pd.read_csv(csv_path)


for col in df.columns:
    if df[col].dtype in ["int64","float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


num_cols = df.select_dtypes(include=["int64","float64"]).columns
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


clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f" Acurácia do modelo: {accuracy:.2f}")


plt.figure(figsize=(40,20))
tree.plot_tree(
    clf,
    feature_names=X_train.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True
)


buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())


