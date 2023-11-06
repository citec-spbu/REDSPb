import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def process_matrix(matrix, threshold,columns_strength):

    rows, cols = matrix.shape

    xs = []

    sss_x = -1
    eee_x = -1
    for j in range(cols):
        count_greater = np.sum(matrix[:, j] >= threshold)

        if count_greater >= rows / columns_strength:
            # for i in range(rows):
            #     if matrix[i, j] < threshold:
            #         matrix[i, j] = 0
            #     else:
            #         matrix[i, j] = 1
            matrix[:, j] = 1
            if sss_x == -1:
                sss_x = j
            eee_x = j
        else:
            matrix[:, j] = 0

            if sss_x != -1:
                xs.append((sss_x, eee_x))
            sss_x = -1
            eee_x = -1

    return matrix, xs



def find_longest_sequence(arr, threshold):
    max_length = 0
    current_length = 0
    start_index = 0
    end_index = 0
    max_start_index = 0

    for i, num in enumerate(arr):
        if num > threshold:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = max_start_index
                end_index = i
        else:
            current_length = 0
            max_start_index = i + 1

    return max_length, start_index, end_index


def vsp(arr: np.ndarray, threshold, coef=1.5, coef2=6):
    res = arr.copy()
    count_greater = np.sum(arr >= threshold)
    if count_greater >= len(arr) / coef:
        res.fill(1)
    else:
        max_length, start_index, end_index = find_longest_sequence(arr, threshold)
        if max_length > coef2:
            res.fill(0)
            for i in range(start_index, end_index + 1):
                res[i] = 1
        else:
            res.fill(1)
    return res


def smart_mark(matrix_raw, matrix_kek, xs, threshold, coef, coef2):
    rows, cols = matrix.shape
    for col_x_s, col_x_e in xs:
        col_x_e += 1
        local_width = col_x_e - col_x_s
        for row in range(rows):
            count_greater = np.sum(matrix_raw[row, col_x_s:col_x_e] >= threshold)

            if count_greater >= local_width / 2:
                matrix_kek[row, col_x_s:col_x_e] = 1
            else:
                matrix_kek[row, col_x_s:col_x_e] = vsp(matrix_raw[row, col_x_s:col_x_e], threshold, coef, coef2)
    return matrix_kek


df = pd.read_csv("/Users/aleksandrallahverdan/Downloads/data1_result_2_half.csv")
df['numbers'] = df['numbers'].map(lambda x: list(map(float, x.split(','))))

matrix = np.array(df['numbers'].values.tolist(), dtype=float)[:, 11650:11750]

column_threshold = -9
columns_strength = 30  # нужно 1/columns_strength чисел больше column_threshold чтобы столбик считался целевым столбиком
row_threshold = -9  # пороговое значение для того чтобы отбирать по строкам в столбиках кто 0 а кто 1
row_coef = 1  # если в широком столбике больше len(столбика) / row_coef чисел больше row_threshold то вся строка берется за 1
row_coef2 = 3  # если в строке столбика меньше row_coef2 чисел больше row_threshold то вся строка берется за 0


new_matrix = matrix.copy()
new_matrix, xs_cols = process_matrix(new_matrix, column_threshold,columns_strength)

new_matrix = smart_mark(matrix, new_matrix, xs_cols, row_threshold, row_coef, row_coef2)

# plt.imshow(new_matrix, cmap='hot')
# plt.show()



plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, подграфик 1
plt.imshow(matrix, cmap='hot')  # График для первого подграфика
plt.title('Subplot 1')  # Заголовок для первого подграфика

# Создание второго подграфика
plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, подграфик 2
plt.imshow(new_matrix, cmap='hot')  # График для второго подграфика
plt.title('Subplot 2')  # Заголовок для второго подграфика

# Отображение графиков
plt.tight_layout()  # Автоматическое расположение подграфиков
plt.show()
