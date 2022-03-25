
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv('2191.csv')

    for i, row in df.iterrows():
        row['start_time']

print(torch.cuda.is_available())

y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# #
# # y_true = np.stack([y_true] * 3)
# # y_scores = np.stack([y_scores] * 3)
print(average_precision_score(y_true, y_scores))
