import pandas as pd

fake = pd.read_csv("data/fake.csv")
true = pd.read_csv("data/true.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)

print("\nFake columns:")
print(fake.columns)

print("\nTrue columns:")
print(true.columns)

print("\nFake head:")
print(fake.head())

print("\nTrue head:")
print(true.head())