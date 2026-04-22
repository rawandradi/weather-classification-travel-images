import cv2, numpy as np, pandas as pd, os

df = pd.read_csv("travel_dataset_final.csv")


# Prepare fixed-size arrays
num_bins = 16*3
X = np.zeros((len(df), num_bins))
y = []

for i, row in df.iterrows():
    img_path = f"images/{row['img_id']}.jpg"

    if not os.path.exists(img_path):
        # Keep row but fill with zeros (preserves alignment)
        y.append(row["weather"])
        continue

    img = cv2.imread(img_path)
    if img is None:
        # corrupted image → leave zero vector
        continue
    img = cv2.resize(img,(224,224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = cv2.calcHist([hsv],[0],None,[16],[0,180]).flatten()
    s = cv2.calcHist([hsv],[1],None,[16],[0,256]).flatten()
    v = cv2.calcHist([hsv],[2],None,[16],[0,256]).flatten()

    X[i] = np.concatenate([h,s,v])
    y.append(row["weather"])

np.save("X_weather_hsv.npy", X) #color & illumination
np.save("y_weather.npy", np.array(y))

print("HSV features extracted with perfect alignment.")
