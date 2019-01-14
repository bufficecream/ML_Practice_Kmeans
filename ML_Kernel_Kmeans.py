import numpy as np 
import datetime
import math
import matplotlib.pyplot as plt

test1_data = open("test1_data.txt", 'r')

test2_data = open("test2_data.txt", 'r')

clusterNum = input("please enter the number of cluster:")

dataXY1 = []
dataXY2 = []

sigma = 8

var = 1 / float(sigma * sigma)
print('1/(sigma ^ 2) : '+str(var))

black = 'k^'

# read points
for line in test1_data:

    datafromtest1 = line.split(" ")

    dataXY1.append((float(datafromtest1[0]),float(datafromtest1[1])))

# print('dataXY1 : ', dataXY1)

for line in test2_data:

    datafromtest2 = line.split(" ")

    dataXY2.append((float(datafromtest2[0]),float(datafromtest2[1])))

# print('dataXY2 : ', dataXY2)


dataXY = dataXY1 #+dataXY2 choose a dataset to run


#prepare for multiclusters
def calculate_centeroid(u_temp):

    countx = 0
    county = 0
    
    for data in u_temp:
        countx = countx + data[0]
        county = county + data[1]

    u_new_x = countx/len(u_temp)
    u_new_y = county/len(u_temp) #[0]

    return (u_new_x, u_new_y)

def showStepData(SteporEnd, stepNum):
    plt.clf()

    for c in range(int(clusterNum)):

        dataToShowX = []
        dataToShowY = []
        for dataxy in u[c]:
            dataToShowX.append(dataxy[0])
            dataToShowY.append(dataxy[1])

        plt.plot(dataToShowX, dataToShowY, 'C'+str(c+1)+'s')
        plt.plot(u_new[c][0], u_new[c][1], black)
                                       
        plt.annotate('C%s' % c, xy=(u_new[c][0], u_new[c][1]), textcoords='data') 

    if 'step' in SteporEnd:
        if stepNum == 0:
            title = SteporEnd+str(stepNum)+'(initial clustering)'
        else:
            title = SteporEnd+str(stepNum)
        plt.title(title)
    else:
        plt.title(SteporEnd)

    plt.draw()
    plt.pause(1)

# RBF Kernel
def rbfKernel(dist_sq):
    return math.exp(dist_sq * (-0.5) * var)


dataToShowX = []
dataToShowY = []
for dataxy in dataXY:
    dataToShowX.append(dataxy[0])
    dataToShowY.append(dataxy[1])

plt.title('Initial')
plt.plot(dataToShowX, dataToShowY, 'C1s')
plt.draw()
plt.pause(1)


#-----CCIA------- initialize the center of the cluster with the first point of two files
u_old = []

for index in range(int(clusterNum)):
    u_old.append(dataXY[index])


u_new = []

for index in range(int(clusterNum)): #default
    u_new.append(u_old[index])


#initial clustering for RBF Kernel by comparing the squared Euclidean distance
u = [[] for _ in range(int(clusterNum))]

for data in dataXY: 
    u_dist_sq = [] # for store the dist with each centeroid

    for c in range(int(clusterNum)):
        dist_sq = pow(data[0] - u_old[c][0], 2) + pow(data[1] - u_old[c][1], 2)
        u_dist_sq.append(dist_sq)

    min_dist = 99999
    indexToAppend = 0

    for c in range(int(clusterNum)):
        if min_dist >= u_dist_sq[c]:
            min_dist = u_dist_sq[c]
            indexToAppend = c

    u[indexToAppend].append(data)


# u_new = []
for c in range(int(clusterNum)):
    u_new.append(calculate_centeroid(u[c]))

for c in range(int(clusterNum)):
    print("u", c," : ", u_new[c], " length : ", len(u[c]))


showStepData('step', 0) #initial cluster graph
u_old = []
u_old = u_new


#after the initial clustering do the Kernel Kmeans
stepNum = 0
while True:

    u_temp = [[] for _ in range(int(clusterNum))]


    #accumulate the kernel(xp, xq)
    k_XpXq = [0] * int(clusterNum)
    for c in range(int(clusterNum)):
        for data_in_u1 in u[c]: 
            for data_in_u2 in u[c]: 
                dist = pow(data_in_u1[0] - data_in_u2[0], 2) + pow(data_in_u1[1] - data_in_u2[1], 2)
                k_XpXq[c] = k_XpXq[c] + rbfKernel(dist)

    for c in range(int(clusterNum)):
            k_XpXq[c] = k_XpXq[c] / len(u[c]) 
            k_XpXq[c] = k_XpXq[c] / len(u[c])
            # print("k_XpXq[",c,"] : ", k_XpXq[c])

    #E : classify all samples according to closet
    for data in dataXY:

        u_feature_dist_sq = [] # for store the dist with each centeroid

        for c in range(int(clusterNum)):
            
            #accumulate the kernel(xj, xn)
            k_XjXn = 0
            for data_in_u in u[c]:
                dist = pow(data[0] - data_in_u[0], 2) + pow(data[1] - data_in_u[1], 2)
                k_XjXn = k_XjXn + rbfKernel(dist)

            k_XjXn = k_XjXn / len(u[c])
            # print("k_XjXn : ", k_XjXn)

            feature_dist_sq = 1 - 2 * k_XjXn + k_XpXq[c]

            u_feature_dist_sq.append(feature_dist_sq)


        min_dist = 99999
        indexToAppend = 0

        for c in range(int(clusterNum)):
            if min_dist >= u_feature_dist_sq[c]:
                min_dist = u_feature_dist_sq[c]
                indexToAppend = c

        u_temp[indexToAppend].append(data)

        # countXXX = countXXX + 1

    #M : re-compute as the mean μk of the points in cluster Ck for k=1,...,K
    u_new = []

    for c in range(int(clusterNum)):
        u_new.append(calculate_centeroid(u_temp[c]))

    for c in range(int(clusterNum)):
        print("temp u", c," : ", u_new[c])


    print("---------")


    stopOrNot_List = []

    for c in range(int(clusterNum)):
        stopOrNot_List.append(abs(u_new[c][0] - u_old[c][0]) < 0.005 or abs(u_new[c][1] - u_old[c][1]) < 0.005)

    stopOrNot = True

    for TF in stopOrNot_List:
        stopOrNot = stopOrNot and TF

    stepNum = stepNum + 1
    if stopOrNot:
        showStepData('End', 0)
        break
    else:
        showStepData('step', stepNum)
        u_old = []
        u_old = u_new
        u = u_temp


for c in range(int(clusterNum)):
    print("u", c," : ", u_new[c])

plt.show()



