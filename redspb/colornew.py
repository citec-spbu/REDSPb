import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

def create_match_matrices(start_size, end_size):
    match_matrices = []  
    for size in range(start_size, end_size - 1, -1):  
        m = np.ones((1, size), dtype=int)
        match_matrices.append(m)  
    return match_matrices

def find_matches(big_matrix, match_matrix):
    match_size = match_matrix.shape[1]
    matches = []
    for start_col in range(big_matrix.shape[1] - match_size + 1):
        if np.array_equal(big_matrix[0, start_col:start_col + match_size], match_matrix[0]):
            matches.append((start_col, start_col + match_size - 1))
    return matches

def create_color_palette():
    colors = ['black', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
    return ListedColormap(colors[:len(colors)])

def scan_and_color_modified(big_matrix, match_matrix):
    rows, cols = big_matrix.shape
    colored_matrix = np.zeros((rows, cols))
    color_palette = create_color_palette()
    current_color_index = 1

    for match_matrix in match_matrices:
        matches = find_matches(big_matrix[0:1, :], match_matrix)
        change_color = False  # mark for color change

        for start_col, end_col in matches:
            if np.all(colored_matrix[0, start_col:end_col + 1] == 0):  # Check if already colored
                colored_matrix[0, start_col:end_col + 1] = current_color_index + 1
                prev_col_range = (start_col, end_col)

                for row in range(1, rows):
                    # Calculate new column range based on previous range expanded by 1.3
                    new_start_col = max(0, int(prev_col_range[0] - (prev_col_range[1] - prev_col_range[0]) * 0.15))
                    new_end_col = min(cols - 1, int(prev_col_range[1] + (prev_col_range[1] - prev_col_range[0]) * 0.15))

                    # Count number of ones in the new range
                    ones_count = np.sum(big_matrix[row, new_start_col:new_end_col + 1])

                    # Check if ones count is within 80% to 130% of the previous range size
                    if not (0.8 * (prev_col_range[1] - prev_col_range[0] + 1) <= ones_count <= 1.3 * (prev_col_range[1] - prev_col_range[0] + 1)):
                        change_color = True  # Update mark to indicate the need for a color change

                    # Only color where there are ones
                    for col in range(new_start_col, new_end_col + 1):
                        if big_matrix[row, col] == 1 and colored_matrix[row, col] == 0:
                            colored_matrix[row, col] = current_color_index + 1

                    # Find the first and last occurrence of 1 within the new range
                    row_slice = big_matrix[row, new_start_col:new_end_col + 1]
                    ones_indices = np.where(row_slice == 1)[0]
                    if ones_indices.size > 0:
                        first_one = ones_indices[0] + new_start_col
                        last_one = ones_indices[-1] + new_start_col
                        # Update prev_col_range based on the occurrence of ones
                        prev_col_range = (first_one, last_one)

                    # Check if the color index needs to be updated
                    if change_color:
                        current_color_index = (current_color_index + 1) % len(color_palette.colors)
                        if current_color_index == 0:  # skip black
                            current_color_index = 1
                        change_color = False

    return colored_matrix, color_palette

df = pd.read_csv('D:/xuexi/practice/REDSPb-main/redspb/kek.csv')
predict_nn = np.array([row.split(',') for row in df['predict_nn'] ], dtype=float)
t = 0.6
matrix = np.where(predict_nn < t, 0, 1)
match_matrices = create_match_matrices(45, 1)  # Match matrices from 1x45 to 1x1
colored_matrix, color_palette = scan_and_color_modified(matrix, match_matrices)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.imshow(matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
ax1.set_title('Predict')

ax2.imshow(colored_matrix, cmap=color_palette)
ax2.set_title('Colored Matrix')

plt.tight_layout()
plt.show()