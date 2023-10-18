import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def process_matrix(matrix, threshold):

    rows, cols = matrix.shape

    for j in range(cols):
        count_greater = np.sum(matrix[:, j] >= threshold)
      
        if count_greater >= rows / 2:
            for i in range(rows):
                if matrix[i, j] < threshold:
                    matrix[i, j] = threshold
        else: 
            for i in range(rows):
                if matrix[i, j] >= threshold:
                    matrix[i, j] = threshold - 1
    return matrix

def string_to_float_list(s):
        return [float(item) for item in s.split(', ')]

def main():
    file_path = 'D:/xuexi/practice/REDSPb-main/redspb/data1.csv'
    df = pd.read_csv(file_path)
    df['numbers'] = df['numbers'].apply(string_to_float_list)
    matrix = np.array(df['numbers'].values.tolist(), dtype=float)

    new_matrix = matrix.copy()
    threshold = -10
    new_matrix = process_matrix(new_matrix, threshold)
    new_matrix = np.where(new_matrix < threshold, 0, 1)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].imshow(matrix, cmap='hot')
    fig.colorbar(axs[0].images[0], ax=axs[0])
    axs[1].imshow(new_matrix, cmap='hot', vmin=0, vmax=1, interpolation='none')
    fig.colorbar(axs[1].images[0], ax=axs[1])

    plt.show()

if __name__ == '__main__':
    main()