# collection
import pandas as pd 
df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.info())
print(df.head())

print("\n==========Pembatas Tiap Langkahnya==========\n")

# cleaning 
print(df.isnull().sum())
df = df.drop_duplicates()

import seaborn as sns 
import matplotlib.pyplot as plt 
sns.boxplot(x=df['IPK'])
plt.show()

print("\n==========Pembatas Tiap Langkahnya==========\n")

# Exploratory Data Analysis (EDA)
print(df.describe())
sns.histplot(df['IPK'], bins=10, kde=True)
plt.show()
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.show()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

print("\n==========Pembatas Tiap Langkahnya==========\n")

# Feature Engineering 
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan", index=False)

print("\n==========Pembatas Tiap Langkahnya==========\n")

# splitting Dataset 
from sklearn.model_selection import train_test_split

x = df.drop('Lulus', axis=1)
y = df['Lulus']

x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(x_train.shape, x_val.shape, x_test.shape)