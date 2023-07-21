
import pandas as pd
import numpy as np


a1 = pd.read_csv("The_cell_expression_data.csv", header=None, index_col=None).to_numpy()  
b1 = pd.read_csv("LRI.csv", header=None, index_col=None).to_numpy()  # LRI obtained after filtering


for w in range(0, b1.shape[0]):  
    zhibiao1 = 0
    zhibiao2 = 0
    flag = False
    for x in range(0, a1.shape[0]):  
        if b1[w][0] == a1[x][0]:
            b11 = a1[x]  
            zhibiao1 = 1
        if b1[w][1] == a1[x][0]:
            b12 = a1[x]  
            zhibiao2 = 1
        if zhibiao1 == 1 and zhibiao2 == 1:  
            flag = True
            break
    if not flag:
        continue
    b11 = np.delete(b11, 0) 
    b12 = np.delete(b12, 0)  

    n = -2
    for i in range(0, 7):
        n = n + 2
        m = 0
        for j in range(0, 7):

            Cheng = b11[n+1] / b11[n] * b12[m+1] / b12[m]

            m = m + 2
            with open(r"melanoma\cell_expression\\" + str(i) + str(j) + ".csv", mode="a") as f:
                f.write(str(Cheng))
                f.write('\n')
            f.close()

print('--------------------------------------------------')

for i in range(0, 7):
    for j in range(0, 7):
        a1 = pd.read_csv(r"melanoma\cell_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # Obtain the result
        SUM = a1.sum()
        with open(r"melanoma\cell_expression\\result.csv", mode="a") as f:  
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(SUM))
            f.write('\n')
        f.close()
explain = "The result.csv file indicates that 0 represents the Melanoma cancer cells,\n1 represents the T cells,\n2 represents the B cells,\n3 represents the Macrophages,\n4 represents the Endothelial cells,\n5 represents the CAFs,\n6 represents the NK cells.\nFor example: 12_xxx represents the communication between T cells and B cells, and xxx is the calculated communication strength."
print('------------------End of calculation---------------------------')
print(explain)
