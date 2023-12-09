import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def generate_colors(n):
    np.random.seed(0) 
    return [np.random.randint(0, 256, 3).tolist() for _ in range(n)]

def match_and_color(matrix, max_pattern_size=50):
    nrows, ncols = matrix.shape
    colored_matrix = np.zeros((nrows, ncols, 3), dtype=np.uint8)  

    colors = generate_colors(max_pattern_size)

    for size in range(1, max_pattern_size + 1):
        pattern = np.ones((4, size), dtype=np.uint8)
        method = cv2.TM_SQDIFF
        res = cv2.matchTemplate(matrix.astype(np.uint8), pattern, method)
        threshold = 0.01
        loc = np.where(res <= threshold)
        for pt in zip(*loc[::-1]):  
            cv2.rectangle(colored_matrix, pt, (pt[0] + size - 1, pt[1] + 4), colors[size - 1], -1)

    return colored_matrix, colors

df = pd.read_csv('D:/xuexi/practice/REDSPb-main/redspb/kek.csv')
predict_nn = np.array([row.split(',') for row in df['predict_nn'] ], dtype=float)
t = 0.6
matrix = np.where(predict_nn < t, 0, 1)
colored_matrix, colors = match_and_color(matrix)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.imshow(matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
ax1.set_title('Predict')
ax2.imshow(colored_matrix)
ax2.set_title('Colored Matrix')

# legend_handles = [mpatches.Patch(color=np.array(color)/255, label=f'{1}x{size}') for size, color in enumerate(colors, start=1)]
# plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=7)  # Adjusted ncol for visibility
plt.tight_layout()
plt.show()