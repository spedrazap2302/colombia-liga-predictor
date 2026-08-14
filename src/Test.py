from load_data import load_data
played, _ = load_data()

print(played["season"].dtype)
print(played["season"].unique())