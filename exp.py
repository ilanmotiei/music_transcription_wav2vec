
import pandas as pd
import torch


if __name__ == "__main__":
    df = pd.read_csv('2191.csv')

    for i, row in df.iterrows():
        row['start_time']

print(torch.cuda.is_available())
