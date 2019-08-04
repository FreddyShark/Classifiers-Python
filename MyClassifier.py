import sys
import math


def NB(trainingSet, testSet):
    """ runs Naive Bayes algorithm on provided training and test set. """

    # num of training samples
    z = len(trainingSet)
    # num of attributes in a sample including class
    n = len(trainingSet[0])

    def bayes(testSample, Hclass):
        """ returns the numerator value of the fractional probability P(Hclass) of
            testSample being classified as hypothesized Hclass given trainingSet."""
        # accumilative probability (numerator of Bayes therom)
        P = 1.0
        # total number of training samples with class = Hclass
        Hclass_cnt = 0
        # calculate P(E|Hclass) for attribute value E
        for i in range(n-1):
            # members of training set which have class = Hclass for this attribute
            Hset = []
            Etest = float(testSample[i])
            # using each training sample in trainingSet
            for x in range(z):
                trainingSample = trainingSet[x]
                Etrain = float(trainingSample[i])
                if trainingSample[n-1] == Hclass:
                    Hset.append(Etrain*10000)
            P *= normalProbability(Etest*10000, Hset)
            Hclass_cnt = len(Hset)
        
        return P*(Hclass_cnt / z)
    
    def normalProbability(Etest, Hset):
        """ returns the probability that a sample attribute has value E if it is in class Hclass
            according to the normal probabliity density function.
            Hset is the set of all values of E that are coupled with a categorization of Hclass. """
        if not Hset:
            print("training data has only one class.")
            return

        # calc mean
        Esum = 0
        for E in Hset:
            Esum += E 
        mean = Esum / len(Hset)

        # calc standard deviation
        numerator = 0
        for E in Hset:
            numerator += (E - mean)**2
        sd = math.sqrt(numerator / (len(Hset) - 1))

        # calc exponent
        power = (Etest - mean)**2 / (2*(sd**2))
        # calc base
        const = 1 / (sd*math.sqrt(2*math.pi))

        return const*math.exp(-power)
        

    # start algorithm. Apply for each test sample provided
    for testSample in testSet:
        Pyes = bayes(testSample, 'yes')
        Pno = bayes(testSample, 'no')
        # categorize
        if Pyes >= Pno:
            print('yes')
        else:
            print('no')


def KNN(trainingSet, testSet, k):
    """ runs k nearest neighbour algorithm on provided training and test set.""" 

    # num of neighbours
    k = int(k)
    # num of training samples
    z = len(trainingSet)
    # num of attributes in a sample including class
    n = len(trainingSet[0])

    def euclidian(trainingSample, testSample):
        """ returns euclidian distance between a single training sample and
            adn test sample based on attribute values. """
        Dsqr = 0
        for i in range(n-1):
            a = float(trainingSample[i])
            b = float(testSample[i])
            Dsqr += (a - b)**2
        return math.sqrt(Dsqr)
    
    def byDiff(indexedDiff):
        """ provides key for list sorting order. """
        return indexedDiff[0]


    # start algorithm. Apply for each test sample provided
    for testSample in testSet:
        indexedDiffs = []
        # for each training sample provided
        for i in range(z):
            dist = euclidian(trainingSet[i], testSample)
            indexedDiffs.append((dist, i))
        
        indexedDiffs.sort(key=byDiff)
        yesCnt = 0
        for x in range(k):
            i = indexedDiffs[x][1]
            # inspect class value of corresponding training samples.
            if trainingSet[i][n-1] == 'yes':
                yesCnt += 1
            
        # categorize test sample
        if yesCnt >= math.ceil(k/2):
            print('yes')
        else:
            print('no')


def extractData(trainingPath, testingPath):
    """ takes file path to training data set and testing data set and returns the 
        data in 2 arrays containing arrays of attributes [[]] [[]]"""
    training = []
    testing = []

    file = open(trainingPath, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        data = line.strip('\n')
        sample = data.split(',')
        training.append(sample)

    file = open(testingPath, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        data = line.strip('\n')
        sample = data.split(',')
        testing.append(sample)

    return training, testing


def main():
    trainingPath = str(sys.argv[1])
    testingPath = str(sys.argv[2])
    algorithm = str(sys.argv[3])

    training, testing = extractData(trainingPath, testingPath)

    if algorithm == 'NB':
        NB(training, testing)

    else:
        characters = len(algorithm)
        if algorithm[characters-2:] == "NN":
            KNN(training, testing, algorithm[:characters-2])


main()
