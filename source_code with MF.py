import csv
import numpy as np

def userItemMatrixAndPairs(file):
    
    # Variables
    maxUserID = -1
    maxItemID = -1

    # First pass to determine the size of the matrix
    with open(file, 'r') as f:
        csvReader = csv.reader(f)
        for row in csvReader:
            userID    = int(row[0]) - 1
            itemID    = int(row[1]) - 1
            maxUserID = max(maxUserID, userID)
            maxItemID = max(maxItemID, itemID)

    # Initialize matrix
    userItemMatrix = np.zeros((maxUserID + 1, maxItemID + 1))

    # Set to store (user, item) pairs
    userItemPairs = set()

    # Second pass to fill the matrix and collect (user, item) pairs
    with open(file, 'r') as f:
        csvReader = csv.reader(f)
        for row in csvReader:
            userID = int(row[0]) - 1
            itemID = int(row[1]) - 1
            rating = float(row[2])

            userItemMatrix[userID, itemID] = rating
            if rating != 0:
                userItemPairs.add((userID + 1, itemID + 1))  # add 1 to convert zero-based index to one-based

    return userItemMatrix, userItemPairs


def SGD(userItemMatrix, pairs, latentFactors, iterations, initialLearnRate, initialRegularization, finalLearnRate, finalRegularization):

    # Variables
    numUsers, numItems = userItemMatrix.shape
    pairs = list(pairs)

    # Randomly initialize user and item vectors
    userVectors = np.random.normal(scale=2./np.sqrt(latentFactors), size=(numUsers, latentFactors))
    itemVectors = np.random.normal(scale=2./np.sqrt(latentFactors), size=(numItems, latentFactors))
    
    # Prepare learning rate decay and regularization growth
    # // learnRateSteps = (finalLearnRate - initialLearnRate) / iterations
    # // currentLearnRate = initialLearnRate
    # // RegularizationSteps = (finalRegularization - initialRegularization) / iterations
    # // currentRegularization = initialRegularization

    # Perform SGD
    for iteration in range(iterations):
        print(f"########## Iteration {iteration + 1}/{iterations} started. ##############")

        np.random.shuffle(pairs)  # Shuffle pairs at the start of each iteration

        for (u, i) in pairs:
            u, i = u - 1, i - 1  

            prediction = np.dot(userVectors[u], itemVectors[i])

            actualRating = userItemMatrix[u, i]
           
            error = actualRating - prediction

            userVectors[u,:] += initialLearnRate * (error * itemVectors[i,:] - initialRegularization * userVectors[u,:])
            itemVectors[i,:] += initialLearnRate * (error * userVectors[u,:] - initialRegularization * itemVectors[i,:])

        # Update learning rate and regularization
        # // currentLearnRate += learnRateSteps
        # // currentRegularization += RegularizationSteps

    return userVectors, itemVectors


def predictRatings(userVectors, itemVectors, testFile):

    # Variables
    predictions = []

    # Open file that does not have ratings
    with open(testFile, 'r') as file:
        csvReader = csv.reader(file)

        for row in csvReader:

            # Extract content from CSV line by line 
            userID = int(row[0]) - 1
            itemID = int(row[1]) - 1
            timeStamp = int(row[2])

            # Get prediction for userID and itemID and round to nearest .5
            prediction = np.dot(userVectors[userID], itemVectors[itemID])
            prediction = round(prediction * 2)/2

            # Default predictions that are less then 0.5
            if (prediction < 0.5 ):
                prediction = 0.5

            # Add list to the preditions list that contains all existing data + prediction 
            predictions.append([userID + 1, itemID + 1, prediction, timeStamp])
            # // print("Prediction: {}".format(prediction))

    return predictions


def writePredictionsToFile(predictions, outputFile):

    with open(outputFile, 'w', newline='') as file:
        csvWriter = csv.writer(file)
        for row in predictions:
            csvWriter.writerow(row)


# Testing Parameters  
latentFactors         = 20 # Previous (100, 40, 20, 20, 20, 20)

initialLearnRate      = 0.002 # Previous (0.001, 0.005, 0.005, 0.005, 0.005, 0.002) 

initialRegularization = 0.01 # Previous (0.1, 0.1, 0.01, 0.01, 0.01, 0.01) 

iterations            = 25 # Previous (50, 100, 50, 50, 100, 25) 

finalLearnRate        = 0.01 # Previous (0.01, 0.01, 0.01) 

finalRegularization   = 0.5 # Previous (0.5, 0.5, 0.5) 

# File Variables 
trainFile  = "20M_train_withratings.csv"
testFile   = "20M_test_withoutratings.csv"
outputFile = "results.csv"

# Using Functions
matrix, pairs = userItemMatrixAndPairs(trainFile)
userVectors, itemVectors = SGD(matrix, pairs, latentFactors, iterations, initialLearnRate, initialRegularization, finalLearnRate, finalRegularization)
predictions = predictRatings(userVectors, itemVectors, testFile)
writePredictionsToFile(predictions, outputFile)

