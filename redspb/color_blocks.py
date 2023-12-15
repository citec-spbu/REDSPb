import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from tqdm import tqdm

# COLOR_PALETTE = [30, 110, 50, 180, 20, 130, 220]
COLOR_PALETTE = [i for i in range(0, 10)]


def process_matrix(matrix):

    rows, cols = matrix.shape

    xs = []

    sss_x = -1
    eee_x = -1
    for j in range(cols):
        count_greater = np.sum(matrix[:, j] >= 0.5)

        if count_greater >= rows / 4:
            # matrix[:, j] = 1
            if sss_x == -1:
                sss_x = j
            eee_x = j
            pass
        else:
            # matrix[:, j] = 0

            if sss_x != -1:
                xs.append((sss_x, eee_x))
            sss_x = -1
            eee_x = -1

    return xs

def do_smth(matrix):
    _COLOR_ITERATOR = 1
    xs = process_matrix(matrix)
    new_matrix = np.zeros(matrix.shape)
    for column_s, column_e in tqdm(xs):
        col_mat = matrix[:, column_s:column_e]
        prev_row_width = 0
        prev_color = COLOR_PALETTE[_COLOR_ITERATOR]
        for row_ind, row in enumerate(col_mat):
            s_p = -1
            e_p = -1
            for pix_ind, pix_value in enumerate(row):
                if pix_value == 1:
                    e_p = pix_ind
                    if s_p == -1:
                        s_p = pix_ind
            if s_p == e_p == -1:
                s_p, e_p = 0, -1
            row_width = e_p - s_p + 1
            if row_width > prev_row_width * 1.6 and abs(row_width - prev_row_width) > 2:
                _COLOR_ITERATOR += 1
                if _COLOR_ITERATOR == len(COLOR_PALETTE): _COLOR_ITERATOR = 1
                prev_color = COLOR_PALETTE[_COLOR_ITERATOR]
                prev_row_width = row_width
            elif row_width < prev_row_width * 0.625 and abs(row_width - prev_row_width) > 2:
                _COLOR_ITERATOR += 1
                if _COLOR_ITERATOR == len(COLOR_PALETTE): _COLOR_ITERATOR = 1
                prev_color = COLOR_PALETTE[_COLOR_ITERATOR]
                prev_row_width = row_width
            else:
                prev_row_width = max(row_width, prev_row_width)  # можно фиксить

            new_matrix[row_ind, (column_s + s_p):(column_s + e_p + 1)] = prev_color

    return new_matrix


df = pd.read_csv('/Users/aleksandrallahverdan/Downloads/redspb_result/data_2_test_predict2.csv')
print('first read done')
df['predict_nn'] = df['predict_nn'].map(lambda x: list(map(lambda x: float(eval(x)[0]), x.split(','))))
print('second read done')
predict_nn = np.array([row for row in df['predict_nn'] ], dtype=float)
t = 0.6
matrix = np.where(predict_nn < t, 0, 1)

new_res_matrix = do_smth(matrix)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.imshow(matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
ax1.set_title('Predict')

colors = ['black', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
фф =  ListedColormap(colors[:len(colors)])

ax2.imshow(new_res_matrix, cmap=фф)
ax2.set_title('Colored Matrix')

plt.tight_layout()
plt.show()