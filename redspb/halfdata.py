import pandas as pd

path_to_file = "D:\practice\data1.csv"
df = pd.read_csv(path_to_file)

split_index = len(df) // 2
df1 = df.iloc[:split_index]
df2 = df.iloc[split_index:]

df1.to_csv("D:\practice\part1.csv", index=False)
df2.to_csv("D:\practice\part2.csv", index=False)
