# -*- coding: utf-8 -*-
'''it returns an array of objects containing group and feature list'''
def getDataFromFile(fileName):
    data = []
    file_in = open(fileName, 'r')

    # read every line
    for instance in file_in.readlines():
        features = [] # create a feature list
        obj = {} # create an object to store group and feature information
        instance = instance.strip() # remove leading and trailing spaces

        values = instance.split(' ') # split row based on spaces
        obj['group'] = float(values[0]) # store first value as group

        # read the remainging line and stores every feature in features list
        for feature in values[1:]:
            if(feature == ''):# omit white space
                continue
            features.append(float(feature))

        obj['features'] = features # add feature list to the object
        data.append(obj)

    return data

'''Normalization using Z-Score'''
import statistics
def normalization(data):
    # making deep copy of data
    normalizedData = []
    for instance in data:
        tempObject = {}
        tempObject['group'] = instance.get('group')
        tempObject['features'] = [feature for feature in instance.get('features')]
        normalizedData.append(tempObject)

    # loop through each features and normalize it
    for i in range(len(normalizedData[0].get('features'))):
        featureList = []

        for instance in normalizedData:
            featureList.append(instance.get('features')[i])

        mean = statistics.mean(featureList)
        stdev = statistics.stdev(featureList)

        for instance in normalizedData:
            feature = instance.get('features')[i]
            feature = (feature - mean) / stdev
            instance['features'][i] = feature

    return normalizedData

import math
'''it accepts array of objects as training data and just list of features for test data'''
def nearestNeighbourClassifier(trainingDataSet, testInstance):
    distances = []
    for trainingInstance in trainingDataSet: #get object from array
        distanceBetweenFeature = float(0)
        # loop through each feature and calculate distance
        for trainingFeature, testingFeature in zip(trainingInstance.get('features'), testInstance):
            distanceBetweenFeature += (trainingFeature - testingFeature)**2

        # store value of euclidean distance with group number as tuples
        euclideanDistance = math.sqrt(distanceBetweenFeature)
        distances.append((euclideanDistance, trainingInstance.get('group')))

    # sort distances and returns the group number
    distances = sorted(distances)
    return distances[0][1]


def leaveOneOutValidator(data):
    correctGuess = 0
    # select each index to leave as test data
    for leaveIndex in range(len(data)):
        testData = data[leaveIndex]
        if(leaveIndex == 0):
            trainingDataSet = data[1:]
        else:
            # add the remaining instances to the training set
            firstSubset = data[0:leaveIndex]
            secondSubset = data[leaveIndex+1:]
            trainingDataSet = firstSubset + secondSubset

        # call nearest neighbour algo for each set and check if it has detected the group correctly
        guessedGroup = nearestNeighbourClassifier(trainingDataSet, testData.get('features'))
        if(guessedGroup == testData.get('group')):
            correctGuess = correctGuess+1

    # returns the overall percentage
    validationScore = correctGuess/len(data)
    return  validationScore * 100

'''it returns a set of features for a given indices'''
def extractFeatures(data, featureIndices):
    newDataSet = []
    for i in range(len(data)):
        obj = {}
        extractedFeatures = []
        for index in featureIndices:
            extractedFeatures.append(data[i].get('features')[index])

        obj['group'] = data[i].get('group')
        obj['features'] = extractedFeatures
        newDataSet.append(obj)

    return newDataSet

def forwardSelection(data):
    print('Beginning Search.\n')
    featureCount = len(data[0].get('features'))

    # indices of features which improves the model
    indices = []
    bestScores = [] # best scores on each iteration

    checkedLocalMaxima = False
    # maximum number of iteration
    for i in range(featureCount):
        best = 0.0
        newIndex = 0

        # add new feature on each iteration for evaluation
        for j in range(featureCount):
            if(j in indices):
                continue

            # copy previously selected indices
            temp = [val for val in indices]
            # add new index
            if(len(temp) != 0):
                if(j < temp[0]):
                    temp.insert(0, j)
                else:
                    temp.append(j)
            else:
                temp.append(j)

            # extract subset of features and groups according to new list of features to evaluate
            newDataSet = extractFeatures(data, temp)
            accuracyRate = leaveOneOutValidator(newDataSet)

            # update accuracy rate on every addition of features
            print('\t Using feature(s)' + str([index+1 for index in temp]) + ' accuracy is ' + str(accuracyRate))
            if(accuracyRate > best):
                best = accuracyRate
                newIndex = j

        # add the index of selected best feature
        indices.append(newIndex)
        indices = sorted(indices)

        # for for local maxima
        if(len(bestScores) != 0 and best < bestScores[len(bestScores) - 1][0]):
            if(checkedLocalMaxima):
                print('\nAddition of features is not improving the model\n')
                bestScores = sorted(bestScores)
                break
            else:
                print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
                checkedLocalMaxima = True
        elif(checkedLocalMaxima):
            checkedLocalMaxima = False

        tempBestScoreIndices = [index+1 for index in indices]
        if(len(tempBestScoreIndices) != featureCount):
            print('\nFeature(s) set' + str(tempBestScoreIndices) + ' was best, accuracy is ' + str(best) + '\n')
        else:
            print()
        bestScores.append((best, tempBestScoreIndices))
        bestScores = sorted(bestScores)


    print('Finished Search!! The best feature subset is ' + str(bestScores[len(bestScores) - 1][1]) + ', which has an accuracy of '+str(bestScores[len(bestScores) - 1][0]))


def backwardElimination(data):
    print('Beginning Search.\n')
    featureCount = len(data[0].get('features'))

    indices = [i for i in range(featureCount)] # indices of features which improves the model

    scoreOfAllFeatures = leaveOneOutValidator(data) #accuracy when all features are selected

    bestScores = scoreOfAllFeatures
    checkedLocalMaxima = False

    # maximum number of iteration
    for i in range(featureCount):
        scoreList = []
        accuracyImproved = False

        # remove a feature on each iteration
        for j in range(len(indices)):
            temp = [index for index in indices] # copy previously selected index list
            temp.pop(j) # remove a feature

            # get features based on the indices of new features
            newDataSet = extractFeatures(data, temp)
            accuracyRate = leaveOneOutValidator(newDataSet)

            # evaluate accuracy and update if needed
            print('\t Using feature(s)' + str([index + 1 for index in temp]) + ' accuracy is ' + str(accuracyRate))
            if (accuracyRate >= bestScores):
                bestScores = accuracyRate
                accuracyImproved = True

            scoreList.append((accuracyRate, temp))

        # check for local maxima
        if(not accuracyImproved):
            if(checkedLocalMaxima):
                print('\nAddition of features is not improving the model\n')
                break
            else:
                print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
                checkedLocalMaxima = True

        scoreList = sorted(scoreList)
        indices = scoreList[len(scoreList) - 1][1]
        print('\nFeature(s) set' + str([index + 1 for index in indices]) + ' was best, accuracy is ' + str(bestScores) + '\n')

    print('Finished Search!! The best feature subset is ' + str([index + 1 for index in indices]) + ', which has an accuracy of ' + str(bestScores))

def main():
    print('Welcome to Calvin Ng\'s Feature Selection Algorithm')
    fileName = input('Type in the name of the file to test : ')

    print('\nType the number of the algorithm you want to run.\n')
    print('1)Forward Selection')
    print('2)Backward Elimination')

    algoType = input('\t\t\t')

    data = getDataFromFile(fileName)

    print('This dataset has '+str(len(data[0].get('features')))+' features (not including the class attribute), with '+str(len(data))+' instances.')

    print('\nPlease wait while I normalize the data…', end=' ')
    normalizedData = normalization(data)
    print('Done!')

    accuracyOfAllFeatures = leaveOneOutValidator(normalizedData)
    print('\nRunning nearest neighbor with all '+str(len(data[0].get('features')))+' features, using “leaving-one-out” evaluation, I get an accuracy of '+str(accuracyOfAllFeatures)+'%\n')

    if(algoType == '1'):
        forwardSelection(normalizedData)
    elif(algoType == '2'):
        backwardElimination(normalizedData)
    else:
        print('Please input correct type')



if __name__ == '__main__':
    main()
