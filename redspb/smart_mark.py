import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler



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


def vsp(arr: np.ndarray, threshold, coef=1.5, coef2=6, coef3=1):
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
            res.fill(coef3)
    return res


def smart_mark(matrix_raw, matrix_kek, xs, threshold, coef, coef2, coef3=1):
    rows, cols = matrix_raw.shape
    for col_x_s, col_x_e in xs:
        col_x_e += 1
        local_width = col_x_e - col_x_s
        for row in range(rows):
            count_greater = np.sum(matrix_raw[row, col_x_s:col_x_e] >= threshold)

            if count_greater >= local_width / 2:
                matrix_kek[row, col_x_s:col_x_e] = 1
            else:
                matrix_kek[row, col_x_s:col_x_e] = vsp(matrix_raw[row, col_x_s:col_x_e], threshold, coef, coef2, coef3)
    return matrix_kek


def get_updated_matrix(inputmatrix, column_threshold, columns_strength, row_threshold, row_coef, row_coef2, row_coef3=1):
    # column_threshold = -9
    # columns_strength = 30  # нужно 1/columns_strength чисел больше column_threshold чтобы столбик считался целевым столбиком
    # row_threshold = -9  # пороговое значение для того чтобы отбирать по строкам в столбиках кто 0 а кто 1
    # row_coef = 1  # если в широком столбике больше len(столбика) / row_coef чисел больше row_threshold то вся строка берется за 1
    # row_coef2 = 3  # если в строке столбика меньше row_coef2 чисел больше row_threshold то вся строка берется за 0

    newmatrix = inputmatrix.copy()
    newmatrix, xs_cols = process_matrix(newmatrix, column_threshold, columns_strength)
    newmatrix = smart_mark(inputmatrix, newmatrix, xs_cols, row_threshold, row_coef, row_coef2, row_coef3)

    return newmatrix


def view_both(inmatrix, newmatrix):
    plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, подграфик 1
    plt.imshow(inmatrix, cmap='hot')  # График для первого подграфика
    plt.title('Subplot 1')  # Заголовок для первого подграфика

    # Создание второго подграфика
    plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, подграфик 2
    plt.imshow(newmatrix, cmap='hot')  # График для второго подграфика
    plt.title('Subplot 2')  # Заголовок для второго подграфика

    # Отображение графиков
    plt.tight_layout()  # Автоматическое расположение подграфиков
    plt.show()


def view_only1(matrix1):
    plt.imshow(matrix1, cmap='hot')
    plt.show()


# df = pd.read_csv("data1_result_2_half.csv")
df = pd.read_csv("data1.csv")
# df = pd.read_csv("data_res_2_half_upd.csv")
df['numbers'] = df['numbers'].map(lambda x: list(map(float, x.split(','))))
matrix = np.array(df['numbers'].values.tolist(), dtype=float)
scaler = MaxAbsScaler()
matrix_numbers = scaler.fit_transform(matrix)
print(matrix.shape)
# my target
# df['target'] = df['target'].map(lambda x: list(map(float, x.split(','))))
# my_matrix = np.array(df['target'].values.tolist(), dtype=float)
my_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
print(my_matrix.shape)
# print(matrix.shape)
# print(my_matrix.shape)
# view_only1(matrix)

my_matrix[:, :2878] = get_updated_matrix(matrix[:, :2878], -5, 30, -5, 2, 1)
my_matrix[:, 2878:2888] = get_updated_matrix(matrix[:, 2878:2888], -9, 30, -11, 2, 1) # done
my_matrix[:, 2888:2918] = get_updated_matrix(matrix[:, 2888:2918], -5, 30, -5, 2, 1) # done
# my_matrix[:, 2918:3003] = get_updated_matrix(matrix[:, 2918:3003], -10, 30, -8, 3, 3) # done
for i in range(len(matrix)): # new
    for j in range(2930, 2931):
        my_matrix[i][j] = 1
for i in range(len(matrix)):# new
    for j in range(2970, 2973):
        my_matrix[i][j] = 1
for i in range(len(matrix)):# new
    for j in range(2999, 3000):
        my_matrix[i][j] = 1
my_matrix[:, 3003:3030] = get_updated_matrix(matrix[:, 3003:3030], -10, 30, -8, 3, 3) # done
for i in range(len(matrix)):# new
    for j in range(3027, 3028):
        my_matrix[i][j] = 1
my_matrix[:, 3040:3058] = get_updated_matrix(matrix[:, 3040:3058], -5, 30, -5, 2, 1) # done
my_matrix[:, 3058:3192] = get_updated_matrix(matrix[:, 3058:3192], -5, 30, -5, 2, 2) # done
my_matrix[:, 3195:3230] = get_updated_matrix(matrix[:, 3195:3230], -8, 30, -5, 2, 2) # done
my_matrix[:, 3236:3282] = get_updated_matrix(matrix[:, 3236:3282], -8, 30, -5, 2, 2) # done
my_matrix[:, 3282:3600] = get_updated_matrix(matrix[:, 3282:3600], -10, 30, -10, 3, 2) # done
my_matrix[:, 4000:4240] = get_updated_matrix(matrix[:, 4000:4240], -18, 20, -19, 1, 1) # done
for i in range(2297, 2547):
    for j in range(4432, 4434):
        my_matrix[i][j] = 1
my_matrix[:, 4440:4505] = get_updated_matrix(matrix[:, 4440:4505], -9, 20, -7, 3, 3) # done
# for i in range(len(matrix)):
#     for j in range(5012, 5013):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(5380, 5381):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(5522, 5523):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(5550, 5551):
        my_matrix[i][j] = 1
my_matrix[:, 5570:5620] = get_updated_matrix(matrix[:, 5570:5620], -10, 30, -9, 3, 1) # done
my_matrix[:, 5620:6000] = get_updated_matrix(matrix[:, 5620:6000], -9, 30, -9, 1, 1) # done
for i in range(len(matrix)):
    for j in range(5720, 5721):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(5777, 5778):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(5804, 5807):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(5918, 5920):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(5975, 5976):
#         my_matrix[i][j] = 1
# my_matrix[:, 6000:6250] = get_updated_matrix(matrix[:, 6000:6250], -17, 30, -18, 2, 1) # done
# for i in range(len(matrix)):
#     for j in range(6089, 6090):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(6203, 6204):
#         my_matrix[i][j] = 1
# my_matrix[:, 6600:6748] = get_updated_matrix(matrix[:, 6600:6748], -15, 30, -15, 3, 3)  # done
for i in range(len(matrix)):
    for j in range(6760, 6765):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(6797, 6802):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(6835, 6839):
        my_matrix[i][j] = 1
my_matrix[:, 6921:7039] = get_updated_matrix(matrix[:, 6921:7039], -5, 30, -5, 3, 1) # done
my_matrix[:, 7039:7100] = get_updated_matrix(matrix[:, 7039:7100], -5, 30, -7, 1.5, 1) # done
# my_matrix[271:510, 7048:7100] = get_updated_matrix(matrix[271:510, 7048:7100], -5, 50, -9, 3, 5)
my_matrix[:, 7400:7460] = get_updated_matrix(matrix[:, 7400:7460], -9, 30, -5, 3, 1) # done
# for i in range(len(matrix)):
#     for j in range(7488, 7489):
#         my_matrix[i][j] = 1
my_matrix[:, 7500:7525] = get_updated_matrix(matrix[:, 7500:7525], -9, 30, -5, 3, 1) # done
# for i in range(len(matrix)):
#     for j in range(7530, 7532):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(7552, 7553):
        my_matrix[i][j] = 1
my_matrix[:, 7560:7600] = get_updated_matrix(matrix[:, 7560:7600], -15, 2, -15, 1, 3) # done
# my_matrix[170:400, 7560:7600] = get_updated_matrix(matrix[170:400, 7560:7600], -15, 2, -15, 0.5, 3)
my_matrix[:, 7623:7627] = get_updated_matrix(matrix[:, 7623:7627], -9, 30, -5, 3, 1) # done
# for i in range(len(matrix)):
#     for j in range(7926, 7927):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(8109, 8110):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(8166, 8167):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(8194, 8196):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(8241, 8242):
#         my_matrix[i][j] = 1
my_matrix[:, 8448:8495] = get_updated_matrix(matrix[:, 8448:8495], -9, 30, -10, 3, 1) # done
# my_matrix[:, 8499:8526] = get_updated_matrix(matrix[:, 8499:8526], -15, 20, -18, 1, 3)

my_matrix[:, 8650:8840] = get_updated_matrix(matrix[:, 8650:8840], -9, 30, -10, 1, 3) # done
for i in range(len(matrix)):
    for j in range(8781, 8800):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(8754, 8755):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(8811, 8812):
        my_matrix[i][j] = 1
my_matrix[:, 8880:8920] = get_updated_matrix(matrix[:, 8880:8920], -9, 30, -9, 3, 1) # done
for i in range(len(matrix)):
    for j in range(8928, 8952):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(8981, 8982):
        my_matrix[i][j] = 1



# Old marking:
for i in range(0, 1633):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
for i in range(0+1633, 671+1633):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
for i in range(671+1633, 677+1633):
    for j in range(9021, 9026):
        my_matrix[i][j] = 1
for i in range(677+1633, 691+1633):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
for i in range(691+1633, 698+1633):
    for j in range(9021, 9026):
        my_matrix[i][j] = 1
for i in range(698+1633, 709+1633):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
for i in range(709+1633, 716+1633):
    for j in range(9021, 9026):
        my_matrix[i][j] = 1
for i in range(716+1633, 730+1633):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
for i in range(730+1633, 737+1633):
    for j in range(9021, 9026):
        my_matrix[i][j] = 1
for i in range(737+1633, 768+1633):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
for i in range(737+1633, 777+1633):
    for j in range(9021, 9026):
        my_matrix[i][j] = 1
for i in range(777+1633, len(matrix)):
    for j in range(9003, 9028):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(9037, 9039):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(9066, 9067): # 9093 9093
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(9094, 9095): # 9093 9093
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(9118, 9140): # 9117
        my_matrix[i][j] = 1
my_matrix[:, 9155:9184] = get_updated_matrix(matrix[:, 9155:9184], -5, 30, -5, 3, 3, 0)# done
my_matrix[:, 9184:9220] = get_updated_matrix(matrix[:, 9184:9220], -9, 30, -9, 1, 3, 0)# done
for i in range(len(matrix)):
    for j in range(9230, 9253): #9229
        my_matrix[i][j] = 1
my_matrix[:, 9220:9260] = get_updated_matrix(matrix[:, 9220:9260], -9, 30, -9, 1, 3) # done

my_matrix[:, 10160:10200] = get_updated_matrix(matrix[:, 10160:10200], -15, 30, -15, 1, 3)# done
# for i in range(len(matrix)):
#     for j in range(10167, 10192):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10254, 10255):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10368, 10369):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10397, 10416): #10395
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10467, 10493): #new
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10480, 10483): #new
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10498, 10503): # 10496
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10535, 10537): # 10534 10538
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10892, 10897): #10891
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10949, 10952):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(10985, 10991):
        my_matrix[i][j] = 1
my_matrix[:, 10945:10955] = get_updated_matrix(matrix[:, 10945:10955], -2, 200, -2, 1,
                                                       1)  # done - лучше не выделилось

for i in range(len(matrix)):
    for j in range(11004, 11010): # 11003
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11054, 11080): # 11053
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(11446, 11469):
#         my_matrix[i][j] = 1
my_matrix[:, 11440:11479] = get_updated_matrix(matrix[:, 11440:11479], -10, 30, -10, 1, 3)# done
for i in range(len(matrix)):
    for j in range(11558, 11582):
        my_matrix[i][j] = 1
my_matrix[:, 11650:11750] = get_updated_matrix(matrix[:, 11650:11750], -9, 30, -9, 1, 3) # done
for i in range(len(matrix)):
    for j in range(11677, 11688):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11755, 11761): #11754
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11774, 11779): #11773 11780
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11838, 11865):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11901, 11902):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11957, 11958):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(11986, 11987):
        my_matrix[i][j] = 1

for i in range(len(matrix)):
    for j in range(12014, 12016):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(12042, 12044):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(12071, 12072):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(12127, 12128):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(12302, 12304):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(12648, 12672):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(12686, 12708):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(12761, 12783):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(12836, 12859):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(12735, 12736):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(12875, 12896):
        my_matrix[i][j] = 1

for i in range(76, 354):
    for j in range(13004, 13007):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(13284, 13310):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(13343, 13367):
#         my_matrix[i][j] = 1
my_matrix[:, 13325:13400] = get_updated_matrix(matrix[:, 13325:13400], -5, 30, -5, 1.5, 2) # done
for i in range(len(matrix)):
    for j in range(13775, 13798):
        my_matrix[i][j] = 1
my_matrix[:, 13880:13920] = get_updated_matrix(matrix[:, 13880:13920], -16, 5, -17, 2, 1) # done слишком бледная видимо

for i in range(len(matrix)):
    for j in range(13999, 14023):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(14260, 14288):
#         my_matrix[i][j] = 1
my_matrix[:, 14240:14300] = get_updated_matrix(matrix[:, 14240:14300], -10, 30, -10, 4, 3) # done слишком бледная видимо
for i in range(len(matrix)):
    for j in range(14461, 14462):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(14935, 14936):
#         my_matrix[i][j] = 1

# for i in range(len(matrix)):
#     for j in range(15072, 15076):
#         my_matrix[i][j] = 1
# my_matrix[:, 15280:15417] = get_updated_matrix(matrix[:, 15280:15417], -17, 30, -17, 1,3)
# for i in range(len(matrix)):
#     for j in range(15296, 15318):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(15334, 15356):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(15372, 15392):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(15399, 15403):
#         my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(15428, 15451):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(15484, 15507): # 15483
        my_matrix[i][j] = 1
my_matrix[:, 15470:15517] = get_updated_matrix(matrix[:, 15470:15517], -10, 30, -10, 1, 3)# done
# for i in range(len(matrix)):
#     for j in range(15522, 15545): #15520
#         my_matrix[i][j] = 1
my_matrix[:, 15517:15553] = get_updated_matrix(matrix[:, 15517:15553], -10, 30, -10, 1,3)# done
for i in range(len(matrix)):
    for j in range(15560, 15582): # 15581
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(15690, 15713): # 15695 15710
        my_matrix[i][j] = 1
my_matrix[:, 15680:15720] = get_updated_matrix(matrix[:, 15680:15720], -10, 30, -11, 1,2)# done

for i in range(len(matrix)):
    for j in range(16094, 16096): # 16093 16097
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(16117, 16119): # 16093 16097
#         my_matrix[i][j] = 1
my_matrix[:, 16200:16250] = get_updated_matrix(matrix[:, 16200:16250], -12, 30, -15, 1, 2)# done
for i in range(len(matrix)):
    for j in range(16510, 16511):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16516, 16542):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16548, 16571): # 16547
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16574, 16597):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16612, 16635): # 16611 16634
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16649, 16672):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16687, 16710): #16686
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16724, 16748):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16762, 16785): # 16761 16784
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16837, 16860): # 16836
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(16906, 16907):
        my_matrix[i][j] = 1
my_matrix[:, 16500:16515] = get_updated_matrix(matrix[:, 16500:16515], -7, 10, -2, 2, 2) # не получилось
my_matrix[:, 16830:16865] = get_updated_matrix(matrix[:, 16830:16865], -9, 10, -9, 10, 3) # не получилось

# for i in range(len(matrix)):
#     for j in range(17580, 17583): # 17737 17761
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(17738, 17760): # 17737 17761
#         my_matrix[i][j] = 1
my_matrix[:, 17725:17770] = get_updated_matrix(matrix[:, 17725:17770], -10, 30, -10, 1, 3)# done
for i in range(len(matrix)):
    for j in range(17813, 17836): # 17812
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(17854, 17855): # 17737 17761
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(17888, 17911):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(17963, 17986):
#         my_matrix[i][j] = 1
my_matrix[:, 17940:17992] = get_updated_matrix(matrix[:, 17940:17992], -10, 30, -10, 1, 3)# done

# for i in range(len(matrix)):
#     for j in range(18024, 18025):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18081, 18082):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18109, 18110):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18137, 18139):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18166, 18167):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18192, 18193):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18526, 18549):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(18959, 18982):
#         my_matrix[i][j] = 1
my_matrix[:, 18950:18990] = get_updated_matrix(matrix[:, 18950:18990], -10, 30, -10, 1, 3)# done
# for i in range(len(matrix)):
#     for j in range(18996, 19020):
#         my_matrix[i][j] = 1
my_matrix[:, 18990:19030] = get_updated_matrix(matrix[:, 18990:19030], -12, 30, -11, 1, 3)# done

for i in range(len(matrix)):
    for j in range(19278, 19302): # 19301
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(19409, 19434): # 19410 19433
#         my_matrix[i][j] = 1
my_matrix[:, 19400:19440] = get_updated_matrix(matrix[:, 19400:19440], -11, 20, -11, 1, 3)# done

for i in range(len(matrix)):
    for j in range(20033, 20034):
        my_matrix[i][j] = 1
for i in range(len(matrix)):
    for j in range(20085, 20109):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(20198, 20221):
#         my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(20312, 20333): # 20311
#         my_matrix[i][j] = 1
my_matrix[:, 20300:20340] = get_updated_matrix(matrix[:, 20300:20340], -10, 30, -10, 3, 3)# done
for i in range(len(matrix)):
    for j in range(20575, 20598):
        my_matrix[i][j] = 1

for i in range(len(matrix)):
    for j in range(21044, 21068):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(21138, 21162):
#         my_matrix[i][j] = 1
my_matrix[:, 21130:21170] = get_updated_matrix(matrix[:, 21130:21170], -5, 30, -5, 1, 3)  # done
for i in range(len(matrix)):
    for j in range(21251, 21273):
        my_matrix[i][j] = 1
# for i in range(len(matrix)):
#     for j in range(21363, 21386):
#         my_matrix[i][j] = 1
my_matrix[:, 21350:21400] = get_updated_matrix(matrix[:, 21350:21400], -5, 30, -5, 3, 3) # done
for i in range(0, 1200):
    for j in range(21542, 21545):
        my_matrix[i][j] = 1

for i in range(len(matrix)):
    for j in range(22264, 22288): # 22286
        my_matrix[i][j] = 1
my_matrix[:, 22485:22515] = get_updated_matrix(matrix[:, 22485:22515], -12, 30, -15, 1, 3) # done
beg = 22000
en = 22540

my_matrix[163:168, :] = 0  # новая полоска
my_matrix[614:615, 6147:8196] = 0
my_matrix[2774, 6147:8195] = 0
my_matrix[1706, 10245:12294] = 0
my_matrix[2779, 10245:12293] = 0

# view_only1(matrix)
# view_both(matrix[0:400, beg:en], my_matrix[0:400, beg:en])
# view_both(matrix[400:800, beg:en], my_matrix[400:800, beg:en])
# view_both(matrix[800:1200, beg:en], my_matrix[800:1200, beg:en])
# view_both(matrix[1200:1600, beg:en], my_matrix[1200:1600, beg:en])
# view_both(matrix[1600:2000, beg:en], my_matrix[1600:2000, beg:en])
# view_both(matrix[2000:2400, beg:en], my_matrix[2000:2400, beg:en])
# view_both(matrix[2400:2800, beg:en], my_matrix[2400:2800, beg:en])
# view_both(matrix[2800:, beg:en], my_matrix[2800:, beg:en])
# view_only1(matrix)
view_both(matrix_numbers, my_matrix)


# Запись в файл
my_matrix = my_matrix.astype(int)
df['target'] = my_matrix.tolist()
df['target'] = df['target'].apply(lambda x: str(x)[1:-1])
df[['date', 'time', 'step1', 'step2', 'step3', 'step4', 'numbers', 'target']].to_csv('data1_marked_experiment.csv')







# сравнение по пикселям с веньда
# df_wenda = pd.read_csv("data1_wenda_1_half.csv")
# # print(df_wenda.columns)
# # df_wenda['target'] = df_wenda['target'].apply(lambda x: str(x)[1:-1])
# df_wenda['target'] = df_wenda['target'].map(lambda x: list(map(float, x.split(','))))
# wenda_matrix = np.array(df_wenda['target'].values.tolist(), dtype=float)
# view_both(wenda_matrix, my_matrix)
