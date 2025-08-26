import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv("constructor_standings.csv")

df = df.loc[:, ~df.columns.duplicated()]

for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


num_cols = df.select_dtypes(include=["int64", "float64"]).columns


scaler_minmax = MinMaxScaler()
df[num_cols] = scaler_minmax.fit_transform(df[num_cols])


print(" Dataset após pré-processamento:")
print(df.head())
print("\n Valores nulos após tratamento:\n", df.isnull().sum())
print("\n Estatísticas descritivas:\n", df.describe(include="all"))