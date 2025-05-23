import numpy as np
import pandas as pd
df = pd.read_csv('Titanic-Dataset.csv')
age = df['Age'].dropna().values  
fare = df['Fare'].dropna().values
print("\nAGE STATISTICS:")
print(f"Mean: {np.mean(age):.1f} years")
print(f"Median: {np.median(age):.1f} years") 
print(f"Std Dev: {np.std(age):.1f} years")

print("\nFARE STATISTICS:")
print(f"Mean: ${np.mean(fare):.2f}")
print(f"Median: ${np.median(fare):.2f}")
print(f"Std Dev: ${np.std(fare):.2f}")
def simple_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
age_norm = simple_normalize(age)
fare_norm = simple_normalize(fare)
print("\nNORMALIZED VALUES (First 5):")
print("Age:", np.round(age_norm[:5], 3))
print("Fare:", np.round(fare_norm[:5], 3))
print("\nVERIFICATION:")
print(f"Age ranges from {np.min(age_norm):.2f} to {np.max(age_norm):.2f}")
print(f"Fare ranges from {np.min(fare_norm):.2f} to {np.max(fare_norm):.2f}")