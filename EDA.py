import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import numpy as np

df = pd.read_csv("travel_dataset_clean_final.csv")
print(df.head())  # shows first 5 rows
print(df.shape)  # shows (number of rows, number of columns)



print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


#correlation for all labels
cat_cols = ["weather", "time_of_day", "season", "activity", "mood", "country"]

for col in cat_cols:
    df[col+"_enc"] = df[col].astype("category").cat.codes

corr_matrix = df[[c+"_enc" for c in cat_cols]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of All Semantic Labels")
plt.tight_layout()
plt.savefig("all_label_correlations.png")
plt.show()

print("\n================ QUANTITATIVE CATEGORICAL STATISTICS ================")

def gini(col):
    p = df[col].value_counts(normalize=True)
    return 1 - np.sum(p**2)

def imbalance_ratio(col):
    c = df[col].value_counts()
    return round(c.max()/c.min(),2)

for col in ["weather","season","time_of_day","activity","mood","country"]:
    print(f"\n--- {col.upper()} ---")
    print("Mode:", df[col].mode()[0])
    print("Entropy:", round(entropy(df[col].value_counts()),3))
    print("Gini Index:", round(gini(col),3))
    print("Imbalance Ratio:", imbalance_ratio(col))
    print("Class Proportions (%):")
    print((df[col].value_counts(normalize=True)*100).round(2))



# Identify rare classes (<5%)
print("\nRare Weather Classes (<5%):")
print((df["weather"].value_counts(normalize=True) * 100)[lambda x: x < 5])

# Visualizations for Weather Label
df["weather"].value_counts().plot(kind="bar", title="Weather Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("weather_distribution.png")
plt.show()


# Weather vs Time of Day
pd.crosstab(df["weather"], df["time_of_day"]).plot(kind="bar", stacked=True)
plt.title("Weather vs Time of Day")
plt.tight_layout()
plt.savefig("weather_time_relation.png")
plt.show()

#Weather vs Season
pd.crosstab(df["season"], df["weather"]).plot(kind="bar", stacked=True)
plt.title("Weather vs Season")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("weather_vs_season.png")
plt.show()

#Weather vs Activity
pd.crosstab(df["activity"], df["weather"]).plot(kind="bar", stacked=True, figsize=(10,5))
plt.title("Weather vs Activity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weather_vs_activity.png")
plt.show()

# Weather vs Mood
pd.crosstab(df["mood"], df["weather"]).plot(kind="bar", stacked=True, figsize=(10,5))
plt.title("Weather vs Mood")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weather_vs_mood.png")
plt.show()


#country vs Weather
pd.crosstab(df["country"], df["weather"]).plot(kind="bar", stacked=True, figsize=(12,6))
plt.title("Country vs Weather")
plt.tight_layout()
plt.savefig("country_vs_weather.png")
plt.show()


print("Final dataset analysis complete.")