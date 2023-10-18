from parser import parse_data 

def cut_data(path_to_file: str):
    row_as_one = 11
    blocks = 3266 # 24 csv
    width_from = 0
    width_to = 22605
    lines_count = blocks * row_as_one 
    names = [f'data{i}.csv' for i in range(1,7)] # всего 24 csv файла, здесь выводятся 6 первых
    lines_from = 0
    for name in names:
        df = parse_data(path_to_file, lines_count, row_as_one, lines_from, width_from, width_to)
        df['numbers'] = df['numbers'].map(lambda x: ', '.join(map(str, x)))
        df.to_csv(name, index=False)
        lines_from += lines_count



# вызов функции:
# path_to_file = 'D:\practice\sdr1\sdr1.txt'
# cut_data(path_to_file)

# чтение из файла:
# df = pd.read_csv("data1.csv")
# df['numbers'] = df['numbers'].map(lambda x: list(map(float, x.split(' '))))


