import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
print(" Arquivos baixados em:", path)


file_path = os.path.join(path, "constructor_standings.csv")
df = pd.read_csv(file_path)


print(" Dimensão do dataset:", df.shape)
print("\n Tipos de variáveis:\n", df.dtypes)
print("\n Resumo estatístico:\n", df.describe(include="all"))
print("\n Valores nulos por coluna:\n", df.isnull().sum())


df.hist(bins=20, figsize=(12, 8))
plt.suptitle("Distribuição das Variáveis Numéricas", fontsize=16)
plt.show()


if len(df.select_dtypes(include="number").columns) > 1:
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação", fontsize=14)
    plt.show()


for col in df.select_dtypes(include="object").columns:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, palette="viridis")
    plt.title(f"Distribuição da variável {col}")
    plt.xticks(rotation=45)
    plt.show()
