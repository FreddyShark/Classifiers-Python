"""
    The following script runs an evaluator on MyClassifier's NB and KNN
    implementations. Will accept path to file and perform 10-fold stratification.
    Results are output to file pima-folds.csv. Uses folds to print the over all
    performance in terms of accuracy, fold accuracies, number of True Positives tp
    and number of False negatives fn, of both algorithms to console.
    To vary k, change its value on line 286
"""
import sys
import math
import copy

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
        

    results = []
    # start algorithm. Apply for each test sample provided
    for testSample in testSet:
        Pyes = bayes(testSample, 'yes')
        Pno = bayes(testSample, 'no')
        # categorize
        if Pyes >= Pno:
            results.append('yes')
        else:
            results.append('no')

    return results


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


    results = []
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
            results.append('yes')
        else:
            results.append('no')
        
    return results



def extractData(trainingPath):
    """ takes file path to training data set and returns the 
        data in a list containing lists of attributes [[], []] . """
    training = []

    file = open(trainingPath, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        data = line.strip('\n')
        sample = data.split(',')
        training.append(sample)

    return training


def stratify10fold(trainingSet):
    """ stratifies the given trainingSet by 10 folds. Returns path to stratified data file
        and list holding the data in its stratified fold lists [[], []]. """
    # num of training samples
    z = len(trainingSet)
    # num of attributes in a sample including class
    n = len(trainingSet[0])
    # num of left over samples when divided into 10 folds
    extrasCnt = z % 10

    yesSet = []
    noSet = []
    stratifiedTraining = []

    for x in range(z):
        trainingSample = trainingSet[x]
        if trainingSample[n-1] == 'yes':
            yesSet.append(trainingSample)
        elif trainingSample[n-1] == 'no':
            noSet.append(trainingSample)
    
    yesCnt = len(yesSet) 
    noCnt = len(noSet)
    yesExtrasCnt = yesCnt % 10
    noExtrasCnt = noCnt % 10

    yesSetExtras = copy.deepcopy(yesSet[ yesCnt - yesExtrasCnt : ])
    noSetExtras = copy.deepcopy(noSet[ noCnt - noExtrasCnt : ])

    file = open('pima-folds.csv', 'w+')
    # distribute among 10 folds evenly
    i, j = 0.0, 0.0
    for x in range(10):
        file.write('fold%d\n' % (x+1))
        fold = []
        while i < (yesCnt - yesExtrasCnt) / 10:
            sample = yesSet.pop(0)
            file.write(','.join(sample))
            file.write('\n')
            fold.append(sample)
            i += 1
        i = 0.0
        while j < (noCnt - noExtrasCnt) / 10:
            sample = noSet.pop(0)
            file.write(','.join(sample))
            file.write('\n')
            fold.append(sample)
            j += 1
        j = 0.0
        # add one from left over from perfect 10 fold division
        if extrasCnt > 0:
            if yesSetExtras:
                sample = yesSetExtras.pop()
                file.write(','.join(sample))
                file.write('\n')
                fold.append(sample)
            elif noSetExtras:
                sample = noSetExtras.pop()
                file.write(','.join(sample))
                file.write('\n')
                fold.append(sample)
            extrasCnt -= 1
        file.write('\n')
        stratifiedTraining.append(fold)
    file.close()
    
    return stratifiedTraining, 'pima-folds.csv'


def evaluatePerformance(stratifiedSet, algorithm):
    """ takes the 10 fold stratified training set and returns a performance measure. """
    # num of attributes
    n = len(stratifiedSet[0][0])

    # holds accuracies obtained using the 9 folds for training and one for testing.
    evals = []
    # True postives count
    tp = 0
    # False negatives count
    fn = 0
    # False Positives count
    fp = 0

    for x in range(10):
        # select xth fold and 
        # remove class categorization to use as test set
        testSetClassified = copy.deepcopy(stratifiedSet[x])
        testSet = [sample[:n-1] for sample in testSetClassified]
        # use other folds as training set (combine them)
        if x == 9:
            training = stratifiedSet[:x]
        else:
            training = stratifiedSet[:x] + stratifiedSet[x+1:]

        trainingSet = [sample for fold in training for sample in fold]

        # run algorithms with formed training and test set
        if algorithm == 'NB':
            classifiedResults = NB(trainingSet, testSet)
        else:
            characters = len(algorithm)
            classifiedResults = KNN(trainingSet, testSet, algorithm[:characters-2])
        
        # count number for True evluations
        correctCnt = 0
        for i in range(len(classifiedResults)):
            if classifiedResults[i] == testSetClassified[i][n-1]:
                correctCnt += 1
                if classifiedResults[i] == 'yes':
                    tp += 1
            else:
                if classifiedResults[i] == 'no':
                    fn += 1
                else:
                    fp += 1
        
        # accuracy based on current test training fold split
        accuracy = correctCnt / len(testSetClassified)
        evals.append(accuracy)
    
    # avg the accuracies obtained
    summation = 0
    for accuracy in evals:
        summation += accuracy
    
    return summation / 10, evals, tp, fn, fp



def main():
    trainingPath = str(sys.argv[1])

    training = extractData(trainingPath)

    stratifiedSet, filedResultsPath = stratify10fold(training)

    ###### set value of k for KNN algorithm ########
    k = 1
    ################################################
    
    algorithmA = str(k) + 'NN'
    performance, foldsPerformance, tp, fn, fp = evaluatePerformance(stratifiedSet, algorithmA)

    print(algorithmA + ' performance: ' + str(performance))
    print('num of True Positives: ' + str(tp))
    print('num of False Positives: ' + str(fp))
    print('num of False Negatives: ' + str(fn))
    print('Fold Accuracies:')
    for x in range(len(foldsPerformance)):
        print(str(x+1) + ': ' + str(foldsPerformance[x]))
    print('\n')

    algorithmB = 'NB'
    performance, foldsPerformance, tp, fn, fp = evaluatePerformance(stratifiedSet, algorithmB)

    print(algorithmB + ' performance: ' + str(performance))
    print('num of True Positives: ' + str(tp))
    print('num of False Positives: ' + str(fp))
    print('num of False Negatives: ' + str(fn))
    print('Fold Accuracies:')
    for x in range(len(foldsPerformance)):
        print(str(x+1) + ': ' + str(foldsPerformance[x]))
    print('\n')
    


main()
