import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def generate_dataset(dataset: pd.DataFrame, n: int = 10, m: int = 100, step_n: int = 10, step_m: int = 10, info=True):
    """
    Create dataset from raw df with sliding window. MinMaxScale numbers
    :param dataset: df
    :param n: height window size
    :param m: width window size
    :param step_n: height window step
    :param step_m: width window step
    :param info: if True -> prints sizes of data in GBs
    :return: pair of scaled X and Y
    """
    matrix_numbers = np.array(dataset['numbers'].values.tolist(), dtype=float)
    matrix_target = np.array(dataset['target'].values.tolist(), dtype=float)

    scaler = MinMaxScaler()
    matrix_numbers = scaler.fit_transform(matrix_numbers)

    if info:
        print('Memory size was:', matrix_numbers.nbytes / 2**30, 'GB, ', matrix_target.nbytes / 2**30, 'GB')

    x = []
    for i in tqdm(range(0, matrix_numbers.shape[0] - n + 1, step_n)):
        for j in range(0, matrix_numbers.shape[1] - m + 1, step_m):
            window = matrix_numbers[i:i + n, j:j + m]
            x.append(window)
    x = np.array(x, dtype=float)

    y = []
    for i in tqdm(range(0, matrix_target.shape[0] - n + 1, step_n)):
        for j in range(0, matrix_target.shape[1] - m + 1, step_m):
            window = matrix_target[i:i + n, j:j + m]
            y.append(window)
    y = np.array(y, dtype=float)

    if info:
        print('Memory size X Y:', x.nbytes / 2**30, 'GB, ', y.nbytes / 2**30, 'GB')
    return x, y


# df = pd.read_csv("/Users/aleksandrallahverdan/Downloads/data1.csv")
# df['numbers'] = df['numbers'].map(lambda x: list(map(float, x[1:-1].split(','))))
# df['target'] = df['target'].map(lambda x: list(map(float, x.split(','))))
#
# X, Y = generate_dataset(df, info=True)
#
# print(X.shape, Y.shape)
