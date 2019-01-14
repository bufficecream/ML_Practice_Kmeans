import numpy as np 
import datetime
import math, time
import matplotlib.pyplot as plt

# plt.ion()

test1_data = open("test1_data.txt", 'r')

test2_data = open("test2_data.txt", 'r')

clusterNum = input("please enter the number of cluster:")

dataXY1 = []
dataXY2 = []

# blue = 'bs'
# red = 'r^'
# green = 'gs'
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


dataXY = dataXY2 #+dataXY2 choose a dataset to run

dataToShowX = []
dataToShowY = []
for dataxy in dataXY:
    dataToShowX.append(dataxy[0])
    dataToShowY.append(dataxy[1])

plt.title('Initial')
plt.plot(dataToShowX, dataToShowY, 'C1s')

plt.draw()
plt.pause(1)

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
        for dataxy in u_temp[c]:
            dataToShowX.append(dataxy[0])
            dataToShowY.append(dataxy[1])

        plt.plot(dataToShowX, dataToShowY, 'C'+str(c+1)+'s')
        plt.plot(u_new[c][0], u_new[c][1], black)
                                      
        plt.annotate('C%s' % c, xy=(u_new[c][0], u_new[c][1]), textcoords='data') 

    if 'step' in SteporEnd:
        plt.title(SteporEnd+str(stepNum))
    else:
        plt.title(SteporEnd)

    plt.draw()
    plt.pause(1)

#-----CCIA------- initialize the center of the cluster with the first point of two files
u_old = []

for index in range(int(clusterNum)):
    u_old.append(dataXY[index])


u_new = []

for index in range(int(clusterNum)): #default
    u_new.append(u_old[index])

stepNum = 0
while True:

    u_temp = [[] for _ in range(int(clusterNum))]

    #E : classify all samples according to closet
    for data in dataXY:

        u_dist_sq = [] # for store the dist with each centeroid

        for c in range(int(clusterNum)):
            dist_sq = pow(data[0] - u_old[c][0], 2) + pow(data[1] - u_old[c][1], 2)
            u_dist_sq.append(dist_sq)
            # u_dist_sq[c]

        min_dist = 99999
        indexToAppend = 0

        for c in range(int(clusterNum)):
            if min_dist >= u_dist_sq[c]:
                min_dist = u_dist_sq[c]
                indexToAppend = c

        u_temp[indexToAppend].append(data)


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

    # print("u_temp length : ", len(u_temp[0]))
    # print("u_temp length : ", len(u_temp[1]))

    stepNum = stepNum + 1
    # plt.close()

    if stopOrNot:
        showStepData('End', 0)
        break
    else:
        showStepData('step', stepNum)
        u_old = []
        u_old = u_new


for c in range(int(clusterNum)):
    print("u", c," : ", u_new[c])

plt.show()




