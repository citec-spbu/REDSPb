import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df1 = pd.read_csv('D:/xuexi/practice/REDSPb-main/redspb/kek.csv')
predict_nn = np.array([row.split(',') for row in df1['predict_nn'] ], dtype=float)
# df2 = pd.read_csv('D:/xuexi/practice/REDSPb-main/redspb/data2.csv')
# numbers = np.array([row.split(',') for row in df2['numbers'] ], dtype=float)
threshold = 0.6
matrix = np.where(predict_nn < threshold, 0, 1)

def process_submatrices(matrix, x, y, t):
    rows, cols = matrix.shape
    change_replacement_rule = False  

    for i in range(0, rows, x):
        for j in range(0, cols, y):
            sub_matrix = matrix[i:i+x, j:j+y]
            count_greater_equal_t = np.sum(sub_matrix >= t)

            half_area = x * y / 2
            four_fifths_area = 4 * x * y / 5

            if half_area <= count_greater_equal_t <= four_fifths_area:
                middle_col = y // 2
                left_count = np.sum(sub_matrix[:, :middle_col] >= t)
                right_count = count_greater_equal_t - left_count

                if left_count < right_count:
                    if not change_replacement_rule:
                        matrix[i:i+x, j:j+middle_col] = t - 1
                        matrix[i:i+x, j+middle_col:j+y] = t
                        change_replacement_rule = True  
                    else:
                        matrix[i:i+x, j:j+y] = t  
                else:
                    matrix[i:i+x, j:j+middle_col] = t
                    matrix[i:i+x, j+middle_col:j+y] = t - 1
                    change_replacement_rule = False  
                continue  

            if count_greater_equal_t < half_area:
                matrix[i:i+x, j:j+y] = t - 1
                change_replacement_rule = False  

            elif count_greater_equal_t > four_fifths_area:
                matrix[i:i+x, j:j+y] = t 
                change_replacement_rule = False  

    matrix1 = np.full((x, 3), t)
    matrix1[:, 1:2] = t - 1

    matrix2 = np.full((x, 4), t)
    matrix2[:, 1:3] = t - 1

    templates = [(matrix1, 3), (matrix2, 4)]

    for i in range(0, rows, x): 
        for j in range(0, cols):  
            for template_matrix, template_cols in templates:  
                if j + template_cols <= cols and i + x <= rows:  
                    sub_matrix = matrix[i:i+x, j:j+template_cols]

                    if np.allclose(sub_matrix, template_matrix, atol=1e-1):
                        matrix[i:i+x, j:j+template_cols] = t
    return matrix


new_matrix = matrix.copy()
x = 7
y = 3
threshold = 1
new_matrix = process_submatrices(new_matrix, x, y, threshold)
 
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

ax1.imshow(matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
ax1.set_title('predict')

ax2.imshow(new_matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
ax2.set_title('threshold = 0.6')

plt.tight_layout()
plt.show()