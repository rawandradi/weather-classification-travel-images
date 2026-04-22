import cv2, numpy as np, pandas as pd, os
from sklearn.cluster import MiniBatchKMeans

df = pd.read_csv("travel_dataset_final.csv")
sift = cv2.SIFT_create()

# -----------------------
# 1) Build visual vocabulary
# -----------------------
all_desc = []

for _, row in df.iterrows():
    img_path = f"images/{row['img_id']}.jpg"
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path,0)
    if img is None:
        continue

    img = cv2.resize(img,(224,224))
    kp, des = sift.detectAndCompute(img,None)
    if des is not None:
        all_desc.append(des)

all_desc = np.vstack(all_desc)

k = 200
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096)
kmeans.fit(all_desc)

np.save("sift_codebook.npy", kmeans.cluster_centers_)

# -----------------------
# 2) Extract aligned features
# -----------------------
X = np.zeros((len(df), k))
y = []

for i, row in df.iterrows():
    img_path = f"images/{row['img_id']}.jpg"
    y.append(row["weather"])

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path,0)
    if img is None:
        continue

    img = cv2.resize(img,(224,224))
    kp, des = sift.detectAndCompute(img,None)

    if des is not None:
        words = kmeans.predict(des)
        for w in words:
            X[i, w] += 1

np.save("X_weather_sift.npy", X) #texture & shape
np.save("y_weather.npy", np.array(y))

print("SIFT BoW features extracted with perfect alignment.")
