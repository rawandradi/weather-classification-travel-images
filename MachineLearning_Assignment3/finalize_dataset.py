import pandas as pd

df = pd.read_csv("travel_dataset_clean_final.csv")

df["img_id"] = range(len(df))

df.to_csv("travel_dataset_clean_final.csv", index=False)
print("Permanent img_id added.")

df = pd.read_csv("travel_dataset_clean_final.csv")

df = df[df["is_image"] == True].reset_index(drop=True)

df.to_csv("travel_dataset_final.csv", index=False)
print("Final dataset created.")

