import csv
import numpy as np

# Files
ratedFile   = "test_100k_rated.csv"
unratedFile = "test_100k_unrated.csv"
outputFile  = "test_results2.csv"

# Reading the rated file
with open(ratedFile, 'r') as file:
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

# Fill rows full of user ratings of items
userItemMatrix = np.zeros((len(uniqueUserIDs), len(uniqueItemIDs)))
for userID, itemID, rating in zip(userIDs, itemIDs, ratings):
    userItemMatrix[userIndex[userID], itemIndex[itemID]] = rating

# Creating similarity matrix
# Calculate the mean rating for each user
sumRatings = np.sum(userItemMatrix, axis=1)
countRatings = np.count_nonzero(userItemMatrix, axis=1)
meanRatings = np.where(countRatings != 0, sumRatings / countRatings, 0)

# Subtract meanRatings from userItemMatrix to adjust for user average
adjustedMatrix = userItemMatrix - meanRatings[:, np.newaxis]

# Initialize the similarity matrix for ITEMS 
similarityMatrix = np.zeros((userItemMatrix.shape[1], userItemMatrix.shape[1]))

# Populate similarity between items in matrix n x n items 
for i in range(similarityMatrix.shape[0]):
    for j in range(similarityMatrix.shape[1]):
        if i != j:
            # Users who rated both items
            commonUsers = np.intersect1d(np.nonzero(userItemMatrix[:, i]), np.nonzero(userItemMatrix[:, j]))
            
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

# Predicting Ratings
predictions = []

# Read the unrated data
with open(unratedFile, 'r') as file:
    csvReader = csv.reader(file)
    for row in csvReader:
        userID = int(row[0])
        itemID = int(row[1])

        u = userIndex[userID]
        i = itemIndex[itemID]

        # Get items that have been rated by user
        itemsRatedByUser = np.where(userItemMatrix[u, :] > 0)[0]

        # Get items that have a positive similarity measure from the index
        neighbourhood = itemsRatedByUser[similarityMatrix[i, itemsRatedByUser] > 0]

        # Calculate the prediction of item-based recommender system using cosine similarity 
        numerator = np.sum(similarityMatrix[i, neighbourhood] * userItemMatrix[u, neighbourhood])
        denominator = np.sum(similarityMatrix[i, neighbourhood])

        # Handle cold cases
        if denominator != 0:
            prediction = numerator / denominator
        else:
            prediction = 0

        predictions.append([userID, itemID, prediction])

# Writing predictions to file
with open(outputFile, 'w', newline='') as file:
    csvWriter = csv.writer(file)
    for row in predictions:
        csvWriter.writerow(row)
