
import pandas as pd



if __name__ == "__main__":
    df = pd.read_csv('2191.csv')

    for i, row in df.iterrows():
        row['start_time']
