import os, pandas as pd, numpy as np

# Load final dataset
df = pd.read_csv("travel_dataset_final.csv")

# Check which rows actually have a downloaded image file
df["has_file"] = df["img_id"].apply(lambda x: os.path.exists(f"images/{x}.jpg"))

print("Real images:", df["has_file"].sum())

# Save real-only dataset
df_real = df[df["has_file"] == True].reset_index(drop=True)
df_real.to_csv("travel_dataset_real.csv", index=False)

# Filter feature matrices
mask = df["has_file"].values

X = np.load("X_weather_final.npy")
y = np.load("y_weather.npy")

X_real = X[mask]
y_real = y[mask]

np.save("X_weather_real.npy", X_real)
np.save("y_weather_real.npy", y_real)

print("Real-only feature set saved.")
