import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math

def generate_colors(n):
    np.random.seed(0) 
    return [np.random.randint(0, 256, 3).tolist() for _ in range(n)]

def match_and_color(matrix, max_pattern_size=30):
    nrows, ncols = matrix.shape
    colored_matrix = np.zeros((nrows, ncols, 3), dtype=np.uint8)
    colors = generate_colors(max_pattern_size + 1)
    color_map = {}

    for size in range(1, max_pattern_size + 1):
        color_index = round(math.log(size, 1.8)) 
        if color_index not in color_map:
            color_map[color_index] = colors[color_index % len(colors)]
        pattern = np.ones((1, size), dtype=np.uint8)
        method = cv2.TM_SQDIFF
        res = cv2.matchTemplate(matrix.astype(np.uint8), pattern, method)
        threshold = 0.01
        loc = np.where(res <= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(colored_matrix, pt, (pt[0] + size - 1, pt[1] + 4), color_map[color_index], -1)
    return colored_matrix, colors

# Это сделать матрицу более прямоугольной.
# def process_submatrices(matrix, x, y, t = 1):
#     rows, cols = matrix.shape
#     change_replacement_rule = False  

#     for i in range(0, rows, x):
#         for j in range(0, cols, y):
#             sub_matrix = matrix[i:i+x, j:j+y]
#             count_greater_equal_t = np.sum(sub_matrix >= t)

#             half_area = x * y / 2
#             four_fifths_area = 4 * x * y / 5

#             if half_area <= count_greater_equal_t <= four_fifths_area:
#                 middle_col = y // 2
#                 left_count = np.sum(sub_matrix[:, :middle_col] >= t)
#                 right_count = count_greater_equal_t - left_count

#                 if left_count < right_count:
#                     if not change_replacement_rule:
#                         matrix[i:i+x, j:j+middle_col] = t - 1
#                         matrix[i:i+x, j+middle_col:j+y] = t
#                         change_replacement_rule = True  
#                     else:
#                         matrix[i:i+x, j:j+y] = t  
#                 else:
#                     matrix[i:i+x, j:j+middle_col] = t
#                     matrix[i:i+x, j+middle_col:j+y] = t - 1
#                     change_replacement_rule = False  
#                 continue  

#             if count_greater_equal_t < half_area:
#                 matrix[i:i+x, j:j+y] = t - 1
#                 change_replacement_rule = False  

#             elif count_greater_equal_t > four_fifths_area:
#                 matrix[i:i+x, j:j+y] = t 
#                 change_replacement_rule = False  

#     matrix1 = np.full((x, 3), t)
#     matrix1[:, 1:2] = t - 1

#     matrix2 = np.full((x, 4), t)
#     matrix2[:, 1:3] = t - 1

#     templates = [(matrix1, 3), (matrix2, 4)]

#     for i in range(0, rows, x): 
#         for j in range(0, cols):  
#             for template_matrix, template_cols in templates:  
#                 if j + template_cols <= cols and i + x <= rows:  
#                     sub_matrix = matrix[i:i+x, j:j+template_cols]

#                     if np.allclose(sub_matrix, template_matrix, atol=1e-1):
#                         matrix[i:i+x, j:j+template_cols] = t
#     return matrix

df = pd.read_csv('D:/xuexi/practice/REDSPb-main/redspb/kek.csv')
predict_nn = np.array([row.split(',') for row in df['predict_nn'] ], dtype=float)
t = 0.6
matrix = np.where(predict_nn < t, 0, 1)
# new_matrix = matrix.copy()
# pmatrix = process_submatrices(new_matrix, 7, 3)
colored_matrix, colors = match_and_color(matrix)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.imshow(matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
ax1.set_title('Predict')

ax2.imshow(colored_matrix)
ax2.set_title('Colored Matrix')

plt.tight_layout()
plt.show()