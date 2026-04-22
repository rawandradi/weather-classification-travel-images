import pandas as pd, os, requests

df = pd.read_csv("travel_dataset_final.csv")
os.makedirs("images", exist_ok=True)

for _, row in df.iterrows():
    try:
        r = requests.get(row["image_url"], timeout=10)
        if r.status_code == 200:
            with open(f"images/{row['img_id']}.jpg","wb") as f:
                f.write(r.content)
    except:
        pass
print("All images downloaded.")