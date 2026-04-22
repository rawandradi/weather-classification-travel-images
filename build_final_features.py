import numpy as np

X_hsv  = np.load("X_weather_hsv.npy")
X_sift = np.load("X_weather_sift.npy")

print("HSV shape:", X_hsv.shape)
print("SIFT shape:", X_sift.shape)

# Concatenate feature vectors
X_final = np.hstack([X_hsv, X_sift])

np.save("X_weather_final.npy", X_final)

print("Final feature vector built:", X_final.shape)
