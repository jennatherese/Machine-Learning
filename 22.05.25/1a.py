import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
print("first 10 rows:")
print(df.head(10))

print("\ndataset shape:")
print(df.shape)

print("\nsummary statistics:")
print(df.describe(include='all'))

print("\ndataset info:")
print(df.info())