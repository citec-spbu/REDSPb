import csv 
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_data(
    path_to_file: str,
    lines_count: int,
    row_as_one: int,
    lines_from: int = 0,
    width_from: int = 0,
    width_to: int = 999999,
):
    """
    :param path_to_file: Path to the data file
    :param lines_count: How many lines to use
    :param row_as_one: How many lines of a file should I take for one
    :param lines_from: How many line to skip from start. Must be a multiple of `row_as_one`
    :param width_from: Data width limit from
    :param width_to: Data width limit to
    :return: pd.DataFrame with parsed data
    """
    df = pd.DataFrame(columns=['date', 'time', 'step1', 'step2', 'step3', 'step4', 'numbers'])
    with open(path_to_file, 'r') as file:
        assert (lines_from / row_as_one) % 1 == 0, f'lines_from не кратно на {row_as_one}'
        assert (lines_count / row_as_one) % 1 == 0, f'lines_count не кратно на {row_as_one}'
        for _ in tqdm(range(lines_from), desc='skipping'):
            next(file)

        pbar = tqdm(total=lines_count, desc='parsing')
        while True:
            if lines_count <= 0:
                break
            result_uni_line = np.array([], dtype=float)
            cols = []
            for i in range(row_as_one):
                line = next(file)
                points = line.split(',')
                cols, numbers = points[:6], [points[6:]]
                numbers = np.array(numbers[0], dtype=float)
                result_uni_line = np.append(result_uni_line, numbers)

            lines_count -= row_as_one
            result_uni_line = result_uni_line[width_from:width_to]
            pbar.update(row_as_one)

            df.loc[len(df)] = cols + [result_uni_line]
        pbar.close()

    return df



def cut_data(path_to_file):
    row_as_one = 11
    blocks = 3266 # 24 csv
    width_from = 0
    width_to = 22605
    lines_count = blocks * row_as_one 
    names = ['data1.csv', 'data2.csv', 'data3.csv', 'data4.csv', 'data5.csv', 'data6.csv',
         'data7.csv', 'data8.csv', 'data9.csv', 'data10.csv', 'data11.csv', 'data12.csv',
         'data13.csv', 'data14.csv', 'data15.csv', 'data16.csv', 'data17.csv', 'data18.csv',
         'data19.csv', 'data20.csv', 'data21.csv', 'data22.csv', 'data23.csv', 'data24.csv']
    lines_from = 0
    for i in names:
        df = parse_data(path_to_file, lines_count, row_as_one, lines_from, width_from, width_to)
        df.to_csv(i, index=False)
        lines_from += lines_count




# path_to_file = 'D:\practice\sdr1\sdr1.txt'
# cut_data(path_to_file)



