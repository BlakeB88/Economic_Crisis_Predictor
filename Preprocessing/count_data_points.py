import pandas as pd


file = "final_merged_dataset_copy.csv"
df = pd.read_csv(file)

complete_rows = df.dropna().shape[0]
total_rows = df.shape[0]

print(f"Total rows in file: {total_rows}")
print(f"Rows with all columns filled: {complete_rows}")
print(f"Percentage of complete rows: {complete_rows / total_rows * 100:.2f}%")
