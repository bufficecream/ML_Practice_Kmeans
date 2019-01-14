import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import linalg
import ML_CCIA as CCIA

test1_data = open("test1_data.txt", 'r')

test2_data = open("test2_data.txt", 'r')

clusterNum = input("please enter the number of cluster:")

dataXY1 = []
dataXY2 = []

sigma = 5

var = 1 / float(sigma * sigma)
print('1/(sigma ^ 2) : '+str(var))

# read points
for line in test1_data:

    datafromtest1 = line.split(" ")

    dataXY1.append((float(datafromtest1[0]),float(datafromtest1[1])))

# print('dataXY1 : ', dataXY1)

for line in test2_data:

    datafromtest2 = line.split(" ")

    dataXY2.append((float(datafromtest2[0]),float(datafromtest2[1])))

# print('dataXY2 : ', dataXY2)

dataXY = dataXY1 #+dataXY1 choose a dataset to run


# RBF Kernel
def rbfKernel(dist_sq):
    return math.exp(dist_sq * (-0.5) * var)


similar_matrix_2d_list = [ [] for row in range(len(dataXY))] 

#generate the 2d similar matrix with RBF kernel
dataxy1_index = 0
for dataxy1 in dataXY:

    for dataxy2 in dataXY:
        dist_sq = pow(dataxy1[0] - dataxy2[0], 2) + pow(dataxy1[1] - dataxy2[1], 2)

        similar_matrix_2d_list[dataxy1_index].append(rbfKernel(dist_sq))

    dataxy1_index = dataxy1_index + 1


degree_matrix_2d_list = [ [] for row in range(len(dataXY)) ]

for row in range(len(similar_matrix_2d_list)):
    sumOfRow = 0
    for col in range(len(similar_matrix_2d_list)):
        sumOfRow = sumOfRow + similar_matrix_2d_list[row][col]

    for count in range(len(similar_matrix_2d_list)):
        degree_matrix_2d_list[row].append(sumOfRow)


# convert to numpy for calculating
similar_matrix_2d = np.array(similar_matrix_2d_list) 

degree_matrix_2d = np.array(degree_matrix_2d_list)


#L = D-W
laplacian_matrix_2d = np.subtract(degree_matrix_2d, similar_matrix_2d)


# get eigenvalues and corresponding eigenvectors
e_val, e_vec = linalg.eig(laplacian_matrix_2d)


afterSortIndexList = sorted(range(len(e_val)), key=lambda k: e_val[k])

U_untranpose = []

for num in range(len(afterSortIndexList)):
    if num == 0:
        continue
    if num > int(clusterNum):
        break

    U_untranpose.append(e_vec[num])

U = np.transpose(U_untranpose)


print('U : ', U)


#prepare for multiclusters
def calculate_centeroid(u_temp):

    count = [0 for num in range(int(clusterNum))]
    
    for data in u_temp:
        for index in range(len(data)):
            count[index] = count[index] + data[index]
    
    centroid = ()

    for index in range(len(data)):
        centroid = centroid + (count[index]/len(u_temp),)

    return centroid


#Start Kmeans
#initialize the center of the cluster with the first point of two files
u_old = []

ccia = CCIA.CCIA(dataXY, clusterNum)

for index in range(int(clusterNum)):
    # u_old.append(dataXY[index])
    u_old.append(ccia[index])


u_new = []

for index in range(int(clusterNum)): #default
    u_new.append(u_old[index])


while True:

    u_temp = [[] for _ in range(int(clusterNum))]
    u_temp_index = [[] for _ in range(int(clusterNum))]

    #E : classify all samples according to closet
    data_index = 0
    for data in U:

        u_dist_sq = [] # for store the dist with each centeroid

        for c in range(int(clusterNum)):
            dist_sq = 0
            for dim in range(int(clusterNum)):
                dist_sq = dist_sq + pow(data[dim] - u_old[c][dim], 2) #+ pow(data[1] - u_old[c][1], 2)

            u_dist_sq.append(float(dist_sq))

        min_dist = 9999
        indexToAppend = 0

        for c in range(int(clusterNum)):
            if min_dist >= u_dist_sq[c]:
                min_dist = u_dist_sq[c]
                indexToAppend = c

        u_temp[indexToAppend].append(data.tolist())
        u_temp_index[indexToAppend].append(data_index)

        data_index = data_index + 1


    #M : re-compute as the mean μk of the points in cluster Ck for k=1,...,K
    u_new = []

    for c in range(int(clusterNum)):
        u_new.append(calculate_centeroid(u_temp[c]))

    for c in range(int(clusterNum)):
        print("temp u", c," : ", u_new[c])


    print("---------")


    stopOrNot_List = []

    for c in range(int(clusterNum)):
        dim_TF_accumulater = False
        for dim in range(int(clusterNum)):
            dim_TF_accumulater = dim_TF_accumulater or (abs(u_new[c][dim] - u_old[c][dim]) < 0.005)

        stopOrNot_List.append(dim_TF_accumulater)

    stopOrNot = True

    for TF in stopOrNot_List:
        stopOrNot = stopOrNot and TF

    if stopOrNot:
        break
    else:
        u_old = []
        u_old = u_new


for c in range(int(clusterNum)):
    print("centroid", c," : ", u_new[c])
    print("cluster", c," including index : ", u_temp_index[c])




