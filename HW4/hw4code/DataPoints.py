class DataPoints:
    # -------------------------------------------------------------------
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.isAssignedToCluster = False
    # -------------------------------------------------------------------
    def __key(self):
        return (self.label, self.x, self.y)
    # -------------------------------------------------------------------
    def __eq__(self, other):
        return self.__key() == other.__key()
    # -------------------------------------------------------------------
    def __hash__(self):
        return hash(self.__key())
    # Computes mean of for each cluster
    @staticmethod
    def getMean(clusters, mean):
        for k in range(len(clusters)):
            temp = clusters[k]
            for point in temp:
                mean[k][0] += point.x
                mean[k][1] += point.y
            mean[k][0] /= float(len(temp))
            mean[k][1] /= float(len(temp))
    # Computes std for each cluster
    @staticmethod
    def getStdDeviation(clusters, mean, stddev):
        for k in range(len(clusters)):
            cluster = clusters[k]
            for point in cluster:
                stddev[k][0] += pow(point.x - mean[k][0], 2)
                stddev[k][1] += pow(point.y - mean[k][1], 2)
            stddev[k][0] /= len(cluster)
            stddev[k][1] /= len(cluster)
    # Computes covariance matrix for each cluster
    @staticmethod
    def getCovariance(clusters, mean, stddev, cov):
        for k in range(len(clusters)):
            cov[k][0][0] = stddev[k][0]
            cov[k][1][1] = stddev[k][1]
            cluster = clusters[k]
            for point in cluster:
                cov[k][0][1] += (point.x - mean[k][0]) * (point.y - mean[k][1])
            cov[k][0][1] /= len(cluster)
            cov[k][1][0] = cov[k][0][1]
    # Get groudtruth number of labels in a dataset
    @staticmethod
    def getNoOFLabels(dataSet):
        labels = set()
        for point in dataSet:
            labels.add(point.label)
        return len(labels)
    # write clusting results into .csv file
    @staticmethod
    def writeToFile(noise, clusters, fileName):
        # write clusters to file for plotting
        f = open(fileName, 'w')
        for pt in noise:
            f.write(str(pt.x) + "," + str(pt.y) + ",0" + "\n")
        for w in range(len(clusters)):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str((w + 1)) + "\n")
        f.close()
