import csv
import numpy as np

def createUserItemMatrix(ratedFile):
    with open(ratedFile, 'r') as file:
        csvReader = csv.reader(file)
        data = list(csvReader)
    
    # userID, itemID, rating, timeStamp
    userIDs = [int(row[0]) for row in data]
    itemIDs = [int(row[1]) for row in data]
    ratings = [float(row[2]) for row in data]
    
    uniqueUserIDs = sorted(set(userIDs))
    uniqueItemIDs = sorted(set(itemIDs))
    
    userIndex = {userID: index for index, userID in enumerate(uniqueUserIDs)}
    itemIndex = {itemID: index for index, itemID in enumerate(uniqueItemIDs)}
    
    userItemMatrix = np.zeros((len(uniqueUserIDs), len(uniqueItemIDs)))
    for userID, itemID, rating in zip(userIDs, itemIDs, ratings):
        userItemMatrix[userIndex[userID], itemIndex[itemID]] = rating
    
    return userItemMatrix, userIndex, itemIndex

def similarityMeasureMatrix(userItemMatrix):
    # Calculate the mean rating for each user
    sumRatings = np.sum(np.where(userItemMatrix > 0, userItemMatrix, 0), axis=1)
    countRatings = np.sum(userItemMatrix > 0, axis=1)
    meanRatings = np.divide(sumRatings, countRatings, out=np.zeros_like(sumRatings), where=countRatings!=0)    
    print(meanRatings)

    # Center the ratings for each user by subtracting the mean rating
    centeredMatrix = userItemMatrix - meanRatings[:, np.newaxis]
    
    # Initialize the similarity matrix
    similarityMatrix = np.zeros((userItemMatrix.shape[0], userItemMatrix.shape[0]))
    
    # Compute the similarity between users
    for i in range(userItemMatrix.shape[0]):
        for j in range(i, userItemMatrix.shape[0]):

            bothRatedIndices = np.where(np.logical_and(userItemMatrix[i] > 0, userItemMatrix[j] > 0))[0]            
            if len(bothRatedIndices) == 0:
                continue
            
            ratingsI = centeredMatrix[i, bothRatedIndices]
            ratingsJ = centeredMatrix[j, bothRatedIndices]
            numerator = np.sum(ratingsI * ratingsJ)
            denominator = np.sqrt(np.sum(ratingsI ** 2)) * np.sqrt(np.sum(ratingsJ ** 2))
            
            if denominator != 0:
                similarityMatrix[i, j] = similarityMatrix[j, i] = numerator / denominator
    
    return similarityMatrix

def createPrediction(userItemMatrix, similarityMatrix, userIndex, itemIndex, unratedFile):
    # Calculate the mean rating for each user, excluding zeros
    meanRatings = np.true_divide(userItemMatrix.sum(1), (userItemMatrix != 0).sum(1))
    print(meanRatings)
    
    predictions = []
    
    # Read the unrated data
    with open(unratedFile, 'r') as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            userID = int(row[0])
            itemID = int(row[1])
            timeStamp = int(row[2])  # Assuming timestamp is always given

            u = userIndex[userID]
            i = itemIndex[itemID]
            
            # Proceed only if the rating is missing
            if userItemMatrix[u, i] == 0:
                # Extract the users who rated the item
                usersRated = np.where(userItemMatrix[:, i] > 0)[0]
                
                # Filter out the users with non-positive similarity
                positiveSimilarityUsers = usersRated[similarityMatrix[u, usersRated] > 0]
                
                numerator = np.sum(similarityMatrix[u, positiveSimilarityUsers] * (userItemMatrix[positiveSimilarityUsers, i] - meanRatings[positiveSimilarityUsers]))
                denominator = np.sum(np.abs(similarityMatrix[u, positiveSimilarityUsers]))
                
                # Predict the rating
                prediction = meanRatings[u]
                if denominator != 0:
                    prediction += numerator / denominator
                
                predictions.append([userID, itemID, prediction, timeStamp])
            else:
                # If there's already a rating, use it
                predictions.append([userID, itemID, userItemMatrix[u, i], timeStamp])

    return predictions

def savePredictions(predictions, outputFile):
    # Write the complete data with predictions to a new CSV file
    with open(outputFile, 'w', newline='') as file:
        csvWriter = csv.writer(file)
        for row in predictions:
            csvWriter.writerow(row)

# Files
ratedFile = 'test_30_rated.csv'  
unratedFile = 'test_30_unrated.csv'
outputFile = 'results.csv'

# Processing
userItemMatrix, userIndex, itemIndex = createUserItemMatrix(ratedFile)
print(userItemMatrix)

similarityMatrix = similarityMeasureMatrix(userItemMatrix)
print(similarityMatrix)

predictionList = createPrediction(userItemMatrix, similarityMatrix, userIndex, itemIndex, unratedFile)
print(predictionList)
for prediction in predictionList:
    print(prediction)

savePredictions(predictionList, outputFile)


