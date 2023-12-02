import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tqdm import tqdm


def generate_dataset(dataset: pd.DataFrame, n: int = 10, m: int = 100, step_n: int = 10, step_m: int = 10, info=True, scale: bool = True, given_scaler=None):
    """
    Create dataset from raw df with sliding window. MinMaxScale numbers
    :param dataset: df
    :param n: height window size
    :param m: width window size
    :param step_n: height window step
    :param step_m: width window step
    :param info: if True -> prints sizes of data in GBs
    :return: pair of scaled X and Y and true shape
    """
    matrix_numbers = np.array(dataset['numbers'].values.tolist(), dtype=float)
    matrix_target = np.array(dataset['target'].values.tolist(), dtype=float)
    mkm_0 = matrix_numbers.shape[0]
    mkm_1 = matrix_numbers.shape[1]
    matrix_numbers = matrix_numbers[:-(mkm_0 % n), :-(mkm_1 % m)]
    matrix_target = matrix_target[:-(mkm_0 % n), :-(mkm_1 % m)]

    if scale:
        if given_scaler is not None:
            matrix_numbers = given_scaler.transform(matrix_numbers)
            print(matrix_numbers.min(), matrix_numbers.max())
            scaler = given_scaler
        else:
            scaler = MaxAbsScaler()
            matrix_numbers = scaler.fit_transform(matrix_numbers)
            print(matrix_numbers.min(), matrix_numbers.max())
    if info:
        print('Memory size was:', matrix_numbers.nbytes / 2**30, 'GB, ', matrix_target.nbytes / 2**30, 'GB')

    x = []
    counter_0 = 0
    for i in tqdm(range(0, matrix_numbers.shape[0] - n + 1, step_n)):
        counter_0 += 1
        counter_1 = 0
        for j in range(0, matrix_numbers.shape[1] - m + 1, step_m):
            counter_1 += 1
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
    return x, y, (counter_0, counter_1), scaler if scale else None


# df = pd.read_csv("/Users/aleksandrallahverdan/Downloads/data1.csv")
# df['numbers'] = df['numbers'].map(lambda x: list(map(float, x[1:-1].split(','))))
# df['target'] = df['target'].map(lambda x: list(map(float, x.split(','))))

# X, Y, sh = generate_dataset(df, info=True)

# print(X.shape, Y.shape, sh)
