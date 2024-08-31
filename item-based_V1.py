import csv
import numpy as np
# userID, itemID, rating, timeStamp
# round to nearest 1

# Takes file of rated items and creates a user-item matrix
def userItemMatrix(file):
    with open(file, 'r') as file:
        csvReader = csv.reader(file)
        data = list(csvReader)
    
    # userID, itemID, rating, timeStamp
    userIDs = [int(row[0]) for row in data]
    itemIDs = [int(row[1]) for row in data]
    ratings = [float(row[2]) for row in data]
    
    # users are y-axis, items are x-axis
    uniqueUserIDs = sorted(set(userIDs))
    uniqueItemIDs = sorted(set(itemIDs))
    
    userIndex = {userID: index for index, userID in enumerate(uniqueUserIDs)}
    itemIndex = {itemID: index for index, itemID in enumerate(uniqueItemIDs)}
    
    # rows full of user ratings of items
    userItemMatrix = np.zeros((len(uniqueUserIDs), len(uniqueItemIDs)))
    for userID, itemID, rating in zip(userIDs, itemIDs, ratings):
        userItemMatrix[userIndex[userID], itemIndex[itemID]] = rating
    
    return userItemMatrix, userIndex, itemIndex

# Takes the user-item matrix and creates a item similarity matrix 
def similarityMatrix(userItemMatrix):
    # Calculate the mean rating for each user
    sumRatings = np.sum(userItemMatrix, axis=1)
    countRatings = np.count_nonzero(userItemMatrix, axis=1)
    meanRatings = np.where(countRatings != 0, sumRatings / countRatings, 0)
    print(meanRatings)

    # Subtract meanRatings from userItemMatrix to adjust for user average
    adjustedMatrix = userItemMatrix - meanRatings[:, np.newaxis]
    print(adjustedMatrix)

    # Initialize the similarity matrix for ITEMS 
    similarityMatrix = np.zeros((userItemMatrix.shape[1], userItemMatrix.shape[1]))

    # Populate similarity between items in matrix n x n items 
    for i in range(similarityMatrix.shape[0]):
        for j in range(similarityMatrix.shape[1]):
            if i != j:
                # Users who rated both items
                commonUsers = np.intersect1d(np.nonzero(userItemMatrix[:, i]), np.nonzero(userItemMatrix[:, j]))
                print(commonUsers)
                
                # Compute the numerator: sum of products of adjusted ratings
                numerator = np.sum(adjustedMatrix[commonUsers, i] * adjustedMatrix[commonUsers, j])
                
                # Compute the denominator: product of the square roots of the sum of squares of adjusted ratings
                denominator = np.sqrt(np.sum(adjustedMatrix[commonUsers, i] ** 2)) * np.sqrt(np.sum(adjustedMatrix[commonUsers, j] ** 2))
                
                # Avoid division by zero
                if denominator != 0:
                    similarityMatrix[i, j] = numerator / denominator
                else:
                    similarityMatrix[i, j] = 0
            else:
                # Similarity of item with itself is 1
                similarityMatrix[i, j] = 1
    
    return similarityMatrix
    
# Takes the userItemMatrix, similarityMatrix, userIndex, itemIndex and unratedFile and create array of preditions for unrated items
def predictRatings(userItemMatrix, similarityMatrix, userIndex, itemIndex, unratedFile):
    
    predictions = []
    
    # Read the unrated data
    with open(unratedFile, 'r') as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            # userID, itemID, rating, timeStamp
            userID = int(row[0])
            itemID = int(row[1])
            timeStamp = int(row[2])

            u = userIndex[userID]
            i = itemIndex[itemID]

            #itemsRatedByUser = np.where(userItemMatrix[:, i] > 0)[0]
            #print(userID)
            #print(itemID)
            #print(itemsRatedByUser)

            #neighbourhood = itemsRatedByUser[similarityMatrix[i, itemsRatedByUser] > 0]

            itemsRatedByUser = np.where(userItemMatrix[u, :] > 0)[0]
            print(userID)
            print(itemID)
            print(itemsRatedByUser)

            neighbourhood = itemsRatedByUser[similarityMatrix[i, itemsRatedByUser] > 0]
            print(neighbourhood)

            numerator = np.sum(similarityMatrix[i, neighbourhood] * userItemMatrix[u, neighbourhood])
            denominator =  np.sum(similarityMatrix[i, neighbourhood])

            if denominator != 0:
                prediction = numerator / denominator
            else:
                prediction = 0
            print(prediction)

            predictions.append([userID, itemID, prediction, timeStamp])
        
    return predictions
    
def savePredictions(predictions, outputFile):
    # Write the complete data with predictions to a new CSV file
    with open(outputFile, 'w', newline='') as file:
        csvWriter = csv.writer(file)
        for row in predictions:
            csvWriter.writerow(row)

ratedFile   = "test_80k_rated.csv"
unratedFile = "test_80k_unrated.csv"
outputFile  = "test_results.csv"

uiMatrix, uIndex, iIndex = userItemMatrix(ratedFile)
print(uiMatrix)

sMatrix = similarityMatrix(uiMatrix)
print(sMatrix)

predictions = predictRatings(uiMatrix, sMatrix, iIndex, uIndex, unratedFile)
print("\nAnswers:")
print(predictions)

savePredictions(predictions, outputFile)