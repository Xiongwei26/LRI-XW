import pandas as pd
import numpy as np

# Processing  scRNA-seq data
single_cell = pd.read_csv("GSE72056.csv", header=None, index_col=None).to_numpy()  # scRNA-seq data


cell_type = np.delete(single_cell[0], 0)
single_cell = single_cell[1:]
malignant = [i for i, x in enumerate(cell_type) if x == 0]
T = [i for i, x in enumerate(cell_type) if x == 1]
B = [i for i, x in enumerate(cell_type) if x == 2]
Mac = [i for i, x in enumerate(cell_type) if x == 3]
End = [i for i, x in enumerate(cell_type) if x == 4]
CAF = [i for i, x in enumerate(cell_type) if x == 5]
NK = [i for i, x in enumerate(cell_type) if x == 6]
gene = single_cell[:, [0]]
gene_data = np.delete(single_cell, 0, axis=1)

row_sums = np.sum(gene_data, axis=1)/gene_data.shape[1]
gene_data_float = gene_data.astype(float)
std = np.std(gene_data_float, axis=1)
total = row_sums + std

malignant_means = np.mean(gene_data[:, malignant], axis=1)
new_arr_0 = [1 if x > y else 0 for x, y in zip(malignant_means, total)]
T_means = np.mean(gene_data[:, T], axis=1)
new_arr_1 = [1 if x > y else 0 for x, y in zip(T_means, total)]
B_means = np.mean(gene_data[:, B], axis=1)
new_arr_2 = [1 if x > y else 0 for x, y in zip(B_means, total)]
Mac_means = np.mean(gene_data[:, Mac], axis=1)
new_arr_3 = [1 if x > y else 0 for x, y in zip(Mac_means, total)]
End_means = np.mean(gene_data[:, End], axis=1)
new_arr_4 = [1 if x > y else 0 for x, y in zip(End_means, total)]
CAF_means = np.mean(gene_data[:, CAF], axis=1)
new_arr_5 = [1 if x > y else 0 for x, y in zip(CAF_means, total)]
NK_means = np.mean(gene_data[:, NK], axis=1)
new_arr_6 = [1 if x > y else 0 for x, y in zip(NK_means, total)]
value_thre = np.append(gene, np.expand_dims(new_arr_0, axis=0).T, axis=1)
value_thre = np.append(value_thre, np.expand_dims(new_arr_1, axis=0).T, axis=1)
value_thre = np.append(value_thre, np.expand_dims(new_arr_2, axis=0).T, axis=1)
value_thre = np.append(value_thre, np.expand_dims(new_arr_3, axis=0).T, axis=1)
value_thre = np.append(value_thre, np.expand_dims(new_arr_4, axis=0).T, axis=1)
value_thre = np.append(value_thre, np.expand_dims(new_arr_5, axis=0).T, axis=1)
value_thre = np.append(value_thre, np.expand_dims(new_arr_6, axis=0).T, axis=1)  # 按列添加值
value_thre = pd.DataFrame(value_thre)
value_thre.to_csv('The_expression_thresholding_data.csv', index=False, header=False)
print("-----Threshold data processing completed----")

value_pro = np.append(gene, np.expand_dims(malignant_means, axis=0).T, axis=1)
value_pro = np.append(value_pro, np.expand_dims(T_means, axis=0).T, axis=1)
value_pro = np.append(value_pro, np.expand_dims(B_means, axis=0).T, axis=1)
value_pro = np.append(value_pro, np.expand_dims(Mac_means, axis=0).T, axis=1)
value_pro = np.append(value_pro, np.expand_dims(End_means, axis=0).T, axis=1)
value_pro = np.append(value_pro, np.expand_dims(CAF_means, axis=0).T, axis=1)
value_pro = np.append(value_pro, np.expand_dims(NK_means, axis=0).T, axis=1)  # 按列添加值
value_pro = pd.DataFrame(value_pro)
value_pro.to_csv('The_expression_product_data.csv', index=False, header=False)
print("-----Product data processing completed----")


malignant_than_zero = (gene_data[:, malignant] > 0)
cell_malignant = np.sum(malignant_than_zero, axis=1)/gene_data[:, malignant].shape[1]
T_than_zero = (gene_data[:, T] > 0)
cell_T = np.sum(T_than_zero, axis=1)/gene_data[:, T].shape[1]
B_than_zero = (gene_data[:, B] > 0)
cell_B = np.sum(B_than_zero, axis=1)/gene_data[:, B].shape[1]
Mac_than_zero = (gene_data[:, Mac] > 0)
cell_Mac = np.sum(Mac_than_zero, axis=1)/gene_data[:, Mac].shape[1]
End_than_zero = (gene_data[:, End] > 0)
cell_End = np.sum(End_than_zero, axis=1)/gene_data[:, End].shape[1]
CAF_than_zero = (gene_data[:, CAF] > 0)
cell_CAF = np.sum(CAF_than_zero, axis=1)/gene_data[:, CAF].shape[1]
NK_than_zero = (gene_data[:, NK] > 0)
cell_NK = np.sum(NK_than_zero, axis=1)/gene_data[:, NK].shape[1]
value_cell = np.append(gene, np.expand_dims(cell_malignant, axis=0).T, axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_T, axis=0).T, axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_B, axis=0).T, axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_Mac, axis=0).T, axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_End, axis=0).T, axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_CAF, axis=0).T, axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_NK, axis=0).T, axis=1)
value_cell = pd.DataFrame(value_cell)
value_cell.to_csv('The_cell_expression_data.csv', index=False, header=False)
print("-----Cell data processing completed----")

