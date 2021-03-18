from hw4code.DataPoints import DataPoints
import random
import sys
import math
import pandas as pd

# =======================================================================
def sqrt(n):
    return math.sqrt(n)

# =======================================================================
def getEuclideanDist(x1, y1, x2, y2):
    dist = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
    return dist
# =======================================================================
def compute_purity(clusters,total_points):
    # Calculate purity

    # Create list to store the maximum union number for each output cluster.
    maxLabelCluster = []
    num_clusters = len(clusters)
    # ========================#
    # STRART YOUR CODE HERE  #
    # ========================#
    import numpy as np
    for clus in clusters:
        # clus is a set of DataPoint objects
        labels = np.asarray([point.label for point in clus])
        maxLabelCluster.append(np.max(np.bincount(labels)))
    # ========================#
    #   END YOUR CODE HERE   #
    # ========================#
    purity = 0.0
    for j in range(num_clusters):
        purity += maxLabelCluster[j]
    purity /= total_points
    print("Purity is %.6f" % purity)

# =======================================================================
def compute_NMI(clusters,noOfLabels):
    # Get the NMI matrix first
    nmiMatrix = getNMIMatrix(clusters, noOfLabels)
    # Get the NMI matrix first
    nmi = calcNMI(nmiMatrix)
    print("NMI is %.6f" % nmi)


# =======================================================================
def getNMIMatrix(clusters, noOfLabels):
    # Matrix shape of [num_true_clusters + 1,num_output_clusters + 1] (example under week6's slide page 9)
    nmiMatrix = [[0 for x in range(len(clusters) + 1)] for y in range(noOfLabels + 1)]
    clusterNo = 0
    for cluster in clusters:
        # Create dictionary {true_class_No: Number of shared elements}
        labelCounts = {}
        # ========================#
        # STRART YOUR CODE HERE  #
        # ========================#
        for point in cluster:
            if point.label not in labelCounts:
                labelCounts[point.label] = 1
            else:
                labelCounts[point.label] += 1
        # ========================#
        #   END YOUR CODE HERE   #
        # ========================#
        labelTotal = 0
        labelCounts_sorted = sorted(labelCounts.items(), key=lambda item: item[1], reverse=True)
        for label, val in labelCounts_sorted:
            nmiMatrix[label - 1][clusterNo] = labelCounts[label]
            labelTotal += labelCounts.get(label)
        # Populate last row (row of summation)
        nmiMatrix[noOfLabels][clusterNo] = labelTotal
        clusterNo += 1
        labelCounts.clear()

    # Populate last col (col of summation)
    lastRowCol = 0
    for i in range(noOfLabels):
        totalRow = 0
        for j in range(len(clusters)):
            totalRow += nmiMatrix[i][j]
        lastRowCol += totalRow
        nmiMatrix[i][len(clusters)] = totalRow

    # Total number of datapoints
    nmiMatrix[noOfLabels][len(clusters)] = lastRowCol

    return nmiMatrix

# =======================================================================
def calcNMI(nmiMatrix):
    # Num of true clusters + 1
    row = len(nmiMatrix)
    # Num of output clusters + 1
    col = len(nmiMatrix[0])
    # Total number of datapoints
    N = nmiMatrix[row - 1][col - 1]
    I = 0.0
    HOmega = 0.0
    HC = 0.0

    for i in range(row - 1):
        for j in range(col - 1):
            # Compute the log part of each pair of clusters within I's formula.
            logPart_I = 1.0
            # ========================#
            # STRART YOUR CODE HERE  #
            # ========================#
            logPart_I = float(N) * nmiMatrix[i][j] / (nmiMatrix[-1][j] * nmiMatrix[i][-1])
            # ========================#
            #   END YOUR CODE HERE   #
            # ========================#

            if logPart_I == 0.0:
                continue
            I += (nmiMatrix[i][j] / float(N)) * math.log(float(logPart_I))

        # Compute HOmega
        # ========================#
        # STRART YOUR CODE HERE  #
        # ========================#
        HOmega += -1 * (nmiMatrix[i][-1] / N) * math.log((nmiMatrix[i][-1] / N))
        # ========================#
        #   END YOUR CODE HERE   #
        # ========================#

    #Compute HC
    # ========================#
    # STRART YOUR CODE HERE  #
    # ========================#
    for j in range(col - 1):
        HC += -1 * (nmiMatrix[-1][j] / N) * math.log((nmiMatrix[-1][j] / N))
    # ========================#
    #   END YOUR CODE HERE   #
    # ========================#

    return I / math.sqrt(HC * HOmega)





# =======================================================================
class Centroid:
    # -------------------------------------------------------------------
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # -------------------------------------------------------------------
    def __eq__(self, other):
        if not type(other) is type(self):
            return False
        if other is self:
            return True
        if other is None:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        return True
    # -------------------------------------------------------------------
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result
    # -------------------------------------------------------------------
    def toString(self):
        return "Centroid [x=" + str(self.x) + ", y=" + str(self.y) + "]"
    # -------------------------------------------------------------------
    def __str__(self):
        return self.toString()
    # -------------------------------------------------------------------
    def __repr__(self):
        return self.toString()







# =======================================================================
class KMeans:
    # -------------------------------------------------------------------
    def __init__(self):
        self.K = 0
    # -------------------------------------------------------------------
    def main(self, dataname,isevaluate=False):
        seed = 71
        self.dataname = dataname[5:-4]
        print("\nFor " + self.dataname)
        self.dataSet = self.readDataSet(dataname)
        self.K = DataPoints.getNoOFLabels(self.dataSet)
        random.Random(seed).shuffle(self.dataSet)
        self.kmeans(isevaluate)
    
    # -------------------------------------------------------------------
    def check_dataloader(self,dataname):

        df = pd.read_table(dataname,sep = "\t", header=None, names=['x','y','ground_truth_cluster'])
        print("\nFor " + dataname[5:-4] + ": number of datapoints is %d" % df.shape[0])
        print(df.head(5))


    # -------------------------------------------------------------------
    def kmeans(self,isevaluate=False):
        clusters = []
        k = 0
        while k < self.K:
            cluster = set()
            clusters.append(cluster)
            k += 1
        
        # Initially randomly assign points to clusters
        i = 0
        for point in self.dataSet:
            clusters[i % k].add(point)
            i += 1

        # calculate centroid for clusters
        centroids = []
        for j in range(self.K):
            centroids.append(self.getCentroid(clusters[j]))

        self.reassignClusters(self.dataSet, centroids, clusters)
        
        # continue till converge
        iteration = 0
        while True:
            iteration += 1
            # calculate centroid for clusters
            centroidsNew = []
            for j in range(self.K):
                centroidsNew.append(self.getCentroid(clusters[j]))

            isConverge = False
            for j in range(self.K):
                if centroidsNew[j] != centroids[j]:
                    isConverge = False
                else:
                    isConverge = True
            if isConverge:
                break

            for j in range(self.K):
                clusters[j] = set()

            self.reassignClusters(self.dataSet, centroidsNew, clusters)
            for j in range(self.K):
                centroids[j] = centroidsNew[j]
        print("Iteration :" + str(iteration))

        if isevaluate:
            # Calculate purity and NMI
            compute_purity(clusters, len(self.dataSet))
            compute_NMI(clusters, self.K)

        # write clusters to file for plotting
        f = open("Kmeans_"+ self.dataname + ".csv", "w")
        for w in range(self.K):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            print(centroids[w].toString())
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str(w) + "\n")
        f.close()

    # -------------------------------------------------------------------
    def reassignClusters(self, dataSet, c, clusters):
        # reassign points based on cluster and continue till stable clusters found
        dist = [0.0 for x in range(self.K)]
        for point in dataSet:
            for i in range(self.K):
               dist[i] = getEuclideanDist(point.x, point.y, c[i].x, c[i].y)

            minIndex = self.getMin(dist)
            # assign point to the closest cluster
            # ========================#
            # STRART YOUR CODE HERE  #
            # ========================#
            for clus in clusters:
                if point in clus:
                    clus.remove(point)
            
            clusters[minIndex].add(point)
            # ========================#
            #   END YOUR CODE HERE   #
            # ========================#
    # -------------------------------------------------------------------
    def getMin(self, dist):
        min = sys.maxsize
        minIndex = -1
        for i in range(len(dist)):
            if dist[i] < min:
                min = dist[i]
                minIndex = i
        return minIndex

    # -------------------------------------------------------------------
    def getCentroid(self, cluster):
        # mean of x and mean of y
        cx = 0
        cy = 0
        # ========================#
        # STRART YOUR CODE HERE  #
        # ========================#
        for c in cluster:
            cx += c.x
            cy += c.y
        
#         print(len(cluster))
        cx = cx / len(cluster)
        cy = cy / len(cluster)
        # ========================#
        #   END YOUR CODE HERE   #
        # ========================#
        return Centroid(cx, cy)
    # -------------------------------------------------------------------
    @staticmethod
    def readDataSet(filePath):
        dataSet = []
        with open(filePath) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            points = line.split('\t')
            x = float(points[0])
            y = float(points[1])
            label = int(points[2])
            point = DataPoints(x, y, label)
            dataSet.append(point)
        return dataSet
