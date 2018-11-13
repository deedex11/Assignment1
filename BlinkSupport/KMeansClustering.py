from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class KMeansClustering(object):

    def __init__(self, clusters, iterations):
        self.numClusters = clusters
        self.iterations = iterations
        pass

    def run(self, x, y, xTrainRaw):
        data = np.array(list(zip(x, y)))

        # X coordinates of random centroids
        C_x = np.random.randint(0, np.max(data) - 20, self.numClusters)
        # Y coordinates of random centroids
        C_y = np.random.randint(0, np.max(data) - 20, self.numClusters)
        C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
        print(C)

        # Plotting along with the Centroids
        plt.scatter(x, y, c='#050505', s=7)
        plt.scatter(C_x, C_y, marker='.', s=200, c='g')

        # To store the value of centroids when it updates
        C_old = np.zeros(C.shape)
        # Cluster Lables(0, 1, 2)
        clusters = np.zeros(len(data))
        # Error func. - Distance between new centroids and old centroids
        error = self.dist(C, C_old, None)
        old = []
        distanceToCluster = np.empty(len(data), dtype=object)        
        # Loop will run till the error becomes zero
        iterationCounter = 0
        while iterationCounter < self.iterations:
            # Assigning each value to its closest cluster
            for i in range(len(data)):
                distances = self.dist(data[i], C)
                cluster = np.argmin(distances)
                clusters[i] = cluster

                distanceToCluster[i] = [cluster, min(distances), i] # this will be the cluster its assigned to and the distance of that cluster and the index 

            # Storing the old centroid values
            C_old = deepcopy(C)
            old.append(C_old)
            # Finding the new centroids by taking the average value
            for i in range(self.numClusters):
                points = [data[j] for j in range(len(data)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)
            error = self.dist(C, C_old, None)
            iterationCounter+=1


        # old is [[ 80.009026 136.74992 ] [ 65.308426  95.18435 ][ 45.613457  58.69688 ] [103.564354 192.7218  ]] x 5
        colors = ['y', 'c', 'm', 'r', 'g', 'b']
        fig, ax = plt.subplots()
        for i in range(self.numClusters):
            points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], alpha=0.5)

        ax.scatter(C[:, 0], C[:, 1], marker='.', s=200, c='#050505', )

        # Instead, lets go through each cluster on its own to draw it
        # That means we need to grab the first array entry of each section in old
        centroidCount = 0
        while centroidCount < self.numClusters:
            centroid = []
            i = 0
            while i < 10:
                centroid.append(old[i][centroidCount])
                i+=1
            # now we should have all x and y values of given cluster
            # so lets draw the arrow tof rall of them
            valueIteration = 0
            newCentroid = 1
            while valueIteration < len(centroid):
                if newCentroid < len(centroid):
                    ax.plot(centroid[valueIteration][0], centroid[valueIteration][1], color=colors[centroidCount], marker=".", markersize="10", markeredgecolor='#050505')
                    ax.plot(centroid[newCentroid][0], centroid[newCentroid][1], color=colors[centroidCount], marker=".", markersize="10", markeredgecolor='#050505')
                    ax.arrow(centroid[valueIteration][0], centroid[valueIteration][1], (centroid[newCentroid][0] - centroid[valueIteration][0]), (centroid[newCentroid][1] - centroid[valueIteration][1]), head_width=0.1, head_length=0.1, length_includes_head=True, fc='#050505', ec=colors[centroidCount])
                valueIteration+=1
                newCentroid +=1
            centroidCount+=1


        ax.figure.savefig("outputYGradient.jpg", bbox_inches='tight')

    # Euclidean Distance Caculator
    def dist(self, a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

    def getClosestDistances(self, distanceToCluster, xTrainRaw):

        # So now we have to collect all the clusters and the distances and grab the least one and the index of each one
        clusterDict = {"0": [], "1": [], "2":[], "3": []}
        for distance in distanceToCluster:
            if distance[0] == 0:
                clusterDict["0"].append(distance)
            if distance[0] == 1:
                clusterDict["1"].append(distance)
            if distance[0] == 2:
                clusterDict["2"].append(distance)
            if distance[0] == 3:
                clusterDict["3"].append(distance)

        # Now we find the minimum of the entire cluster for each one!
        min_index_0 = min(clusterDict["0"], key=lambda x: x[1])
        print("min index for cluster 0: " + str(min_index_0))
        print("The feature data is index %s with [%s, %s]" % (min_index_0[2], x[min_index_0[2]], y[min_index_0[2]]))
        print("xTrainRaw data at that index: " + str(xTrainRaw[min_index_0[2]]))
        min_index_1 = min(clusterDict["1"], key=lambda x: x[1])
        print("min index for cluster 1: " + str(min_index_1))
        print("The feature data is index %s with [%s, %s]" % (min_index_1[2], x[min_index_1[2]], y[min_index_1[2]]))
        print("xTrainRaw data at that index: " + str(xTrainRaw[min_index_1[2]]))
        min_index_2 = min(clusterDict["2"], key=lambda x: x[1])
        print("min index for cluster 2: " + str(min_index_2))
        print("The feature data is index %s with [%s, %s]" % (min_index_2[2], x[min_index_2[2]], y[min_index_2[2]]))
        print("xTrainRaw data at that index: " + str(xTrainRaw[min_index_2[2]]))
        min_index_3 = min(clusterDict["3"], key=lambda x: x[1])
        print("min index for cluster 3: " + str(min_index_3))
        print("The feature data is index %s with [%s, %s]" % (min_index_3[2], x[min_index_3[2]], y[min_index_3[2]]))
        print("xTrainRaw data at that index: " + str(xTrainRaw[min_index_3[2]]))
