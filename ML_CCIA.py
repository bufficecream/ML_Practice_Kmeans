import numpy as np 
import datetime
import math, time
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfinv

#prepare for multiclusters
def calculate_centeroid(u_temp, clusterNum):

    count = [0 for num in range(int(clusterNum))]
   
    centroid = ()

    if type(u_temp[0]) != float:

        for data in u_temp:
            for index in range(len(data)):
                count[index] = count[index] + data[index]     
        
        for index in range(len(data)):
            centroid = centroid + (count[index]/len(u_temp),)
    else:
        centroid = 0
        for data in u_temp:
            count[0] = count[0] + data

        centroid = centroid + (count[0]/len(u_temp))

    return centroid


def showStepData(SteporEnd, stepNum, u_temp, u_new, clusterNum):
    plt.clf()

    dataDims_local = len(str(u_temp[0][0]).split(', '))
    # print('dataDims_local : ', dataDims_local)

    for c in range(int(clusterNum)):

        # print('u_temp : ', u_temp)

        dataToShow = [[] for _ in range(dataDims_local)]

        for dataxy in u_temp[c]:
            if dataDims_local > 1:
                for dataDim in range(dataDims_local):
                    dataToShow[dataDim].append(dataxy[dataDim])
            else:
                dataToShow[0].append(dataxy)

        if dataDims_local == 2:
            plt.plot(dataToShow[0], dataToShow[1], 'C'+str(c+1)+'s')
            plt.plot(u_new[c][0], u_new[c][1], black)
                                          
            plt.annotate('C%s' % c, xy=(u_new[c][0], u_new[c][1]), textcoords='data') 


    if dataDims_local == 2:

        if 'step' in SteporEnd:
            plt.title(SteporEnd+str(stepNum))
        else:
            plt.title(SteporEnd)

        plt.draw()
        plt.pause(1)

def Kmeans(dataXY, initial_clusters,clusterNum):
    
    u_old_func = initial_clusters

    u_new_func = []

    for index in range(int(clusterNum)): #default
        u_new_func.append(u_old_func[index])

    stepNum_func = 0
    while True:

        u_temp_func = [[] for _ in range(int(clusterNum))]

        #E : classify all samples according to closet
        for data in dataXY:

            u_dist_sq = [] # for store the dist with each centeroid

            for c in range(int(clusterNum)):

                dist_sq = 0
                for dim in range(int(clusterNum)):
                    if type(data) == float:
                        dist_sq = dist_sq + pow(data - u_old_func[c], 2)
                    else:
                        dist_sq = dist_sq + pow(data[dim] - u_old_func[c][dim], 2) #+ pow(data[1] - u_old_func[c][1], 2)

                u_dist_sq.append(float(dist_sq))
                # u_dist_sq.append(dist_sq)

            min_dist = 99999
            indexToAppend = 0

            for c in range(int(clusterNum)):
                if min_dist >= u_dist_sq[c]:
                    min_dist = u_dist_sq[c]
                    indexToAppend = c

            u_temp_func[indexToAppend].append(data)


        #M : re-compute as the mean μk of the points in cluster Ck for k=1,...,K
        u_new_func = []

        for c in range(int(clusterNum)):
            u_new_func.append(calculate_centeroid(u_temp_func[c], clusterNum))

        # for c in range(int(clusterNum)):
        #     print("temp u_func", c," : ", u_new_func[c])


        print("---------")


        stopOrNot_List_func = []

        for c in range(int(clusterNum)):
            dim_TF_accumulater = False

            dataDims_u_old = len(str(u_old_func[c]).split(', '))

            if dataDims_u_old == 1:
                dim_TF_accumulater = dim_TF_accumulater or (abs(u_new_func[c] - u_old_func[c]) < 0.005)
            else:
                for dim in range(dataDims_u_old): #int(clusterNum)
                    dim_TF_accumulater = dim_TF_accumulater or (abs(u_new_func[c][dim] - u_old_func[c][dim]) < 0.005)

            stopOrNot_List_func.append(dim_TF_accumulater)

        stopOrNot_func = True

        for TF in stopOrNot_List_func:
            stopOrNot_func = stopOrNot_func and TF

        # print("u_temp_func length : ", len(u_temp_func[0]))
        # print("u_temp_func length : ", len(u_temp_func[1]))

        stepNum_func = stepNum_func + 1
        # plt.close()

        if stopOrNot_func:
            showStepData('End', 0, u_temp_func, u_new_func, clusterNum)
            break
        else:
            showStepData('step', stepNum_func, u_temp_func, u_new_func, clusterNum)
            u_old_func = []
            u_old_func = u_new_func


    # for c in range(int(clusterNum)):
    #     print("u_func", c," : ", u_new_func[c])

    # plt.show()
    return u_new_func, u_temp_func


def phiinv(x):
    #Cumulative distribution function for the standard normal distribution'
    return (math.sqrt(2.0) * erfinv(2*x-1))


#-----CCIA------- initialize the center of the cluster with the first point of two files
def CCIA(getdataXY, clusterNum):

    dataXY = getdataXY #+dataXY2 choose a dataset to run

    dataDims = len(str(dataXY[0]).split(', '))


    u_old = []

    u_pattern_string = [[] for _ in range(len(dataXY)) ]

    pattern_dict = {}

    for dim in range(len(dataXY[0])):
        # print('dim : ', dim)

        mean = 0
        std = 0

        dataDim = []
        for dataNum in range(len(dataXY)):
            dataDim.append(dataXY[dataNum][dim]) #tuple for generalizing

        for dataNum in range(len(dataXY)):
            mean = mean + dataXY[dataNum][dim] 

        mean = mean / len(dataXY)


        dim_sum_sq = 0
        for dataNum in range(len(dataXY)):
            dim_sum_sq = dim_sum_sq + pow(mean - dataXY[dataNum][dim], 2)

        std = math.sqrt(dim_sum_sq / len(dataXY))


        Zs = []
        for s in range(1, int(clusterNum)+1):
            cdf = float(2 * s - 1)/(2 * int(clusterNum))
            # print('cdf : ', cdf)
            Zs.append(phiinv(cdf))

        Xs = []
        for s in range(int(clusterNum)):
            # print('Xs : ', Zs[s] * std + mean)
            Xs.append(Zs[s] * std + mean)

        # print(Xs)

        u_attr_centroids, u_attr_cluster = Kmeans(dataDim, Xs, clusterNum)

        pattern_dict[dim+1] = u_attr_centroids


        data_Num_for_pattern = 0 
        for data in dataXY:
            for c in range(len(u_attr_centroids)):
                if data[dim] in u_attr_cluster[c]:
                    u_pattern_string[data_Num_for_pattern].append(str(c+1))

            data_Num_for_pattern = data_Num_for_pattern + 1

    # print('u_pattern_string : ', u_pattern_string)

    # print('pattern_dict : ', pattern_dict)


    get_diff_type_in_pattern_String = []

    for pattern_string in u_pattern_string:
        if pattern_string not in get_diff_type_in_pattern_String:
            get_diff_type_in_pattern_String.append(pattern_string)

    # print('get_diff_type_in_pattern_String : ', get_diff_type_in_pattern_String)

    k_ = len(get_diff_type_in_pattern_String)

    # print('k_ : ', k_)


    all_centroids_k_ = {}


    to_merge_list_before = []
    while k_ > int(clusterNum):

        for pattern in get_diff_type_in_pattern_String:

            centroid_k_ = ()
            for dim in range(len(pattern)):
                centroid_k_ = centroid_k_ + (pattern_dict[dim+1][int(pattern[dim])-1],)

            pattern_tuple = ()
            for element in pattern:
                
                pattern_tuple = pattern_tuple + (element,)

            if pattern_tuple not in to_merge_list_before:
                all_centroids_k_[pattern_tuple] = centroid_k_
        

        # print('all_centroids_k_ : ', all_centroids_k_)

        value_min_dist = 999999
        to_merge_list = []
        for key, value in all_centroids_k_.items():
            for key_, value_ in all_centroids_k_.items():
                if key == key_ and value == value_:
                    break
                else:
                    value_sum_sq = 0
                    for dim in range(len(value)):
                        value_sum_sq = value_sum_sq + pow(float(value[dim]) - float(value_[dim]), 2)
                    if value_min_dist >= value_sum_sq:
                        value_min_dist = value_sum_sq
                        # print('key : ', key)
                        # print('key_ : ', key_)
                        # print('value_sum_sq : ', value_sum_sq)
                        to_merge_list = []
                        to_merge_list.append(key)
                        to_merge_list.append(key_)

        # print('to_merge_list : ', to_merge_list)


        sumx = 0
        sumy = 0
        sumxy_len = 0
        for index in range(len(u_pattern_string)):
            # print('tuple : ', (u_pattern_string[index][0], u_pattern_string[index][1]))
            if (u_pattern_string[index][0], u_pattern_string[index][1]) in to_merge_list:
                sumx = sumx + dataXY[index][0]
                sumy = sumy + dataXY[index][1]
                sumxy_len = sumxy_len + 1


        sumx = sumx / sumxy_len
        sumy = sumy / sumxy_len

        # print('all_centroids_k_ before : ', all_centroids_k_)

        del all_centroids_k_[to_merge_list[0]]
        del all_centroids_k_[to_merge_list[1]]

        # print('all_centroids_k_ after : ', all_centroids_k_)

        all_centroids_k_[(to_merge_list[0], to_merge_list[1])] = (sumx, sumy)

        to_merge_list_before.append(to_merge_list[0])
        to_merge_list_before.append(to_merge_list[1])


        k_ = k_ - 1

    key_index = 0

    # print('all_centroids_k_ : ', all_centroids_k_)

    initialcluster_return = []
    for key, value in all_centroids_k_.items():
        print('initial centroid',key_index,' : ', value)
        initialcluster_return.append(value)
        key_index = key_index + 1

    return initialcluster_return
  



