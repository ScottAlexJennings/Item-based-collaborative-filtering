import csv
import numpy as np

def userItemMatrix(ratedFile):
    ratingsDict = {}
    maxUserID = -1
    maxItemID = -1

    # Reading the rated file
    with open(ratedFile, 'r') as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            # userID, itemID, rating, timeStamp
            userID = int(row[0]) -1
            itemID = int(row[1]) -1 
            rating = float(row[2])
            timeStamp = int(row[3])

            # Update the dictionary
            ratingsDict[(userID, itemID)] = rating

            # Update maxUserID and maxItemID
            maxUserID = max(maxUserID, userID)
            maxItemID = max(maxItemID, itemID)
    
    # Initialize matrix
    userItemMatrix = np.zeros((maxUserID + 1, maxItemID + 1))

    # Fill the matrix using the dictionary
    for (userID, itemID), rating in ratingsDict.items():
        userItemMatrix[userID, itemID] = rating

    return userItemMatrix

# Creating similarity matrix
def similarityMatrix(userItemMatrix):
    
    # Calculate the mean rating for each user
    sumRatings = np.sum(userItemMatrix, axis=1)
    countRatings = np.count_nonzero(userItemMatrix, axis=1)
    meanRatings = np.where(countRatings != 0, sumRatings / countRatings, 0)

    # Subtract meanRatings from userItemMatrix to adjust for user average
    adjustedMatrix = userItemMatrix - meanRatings[:, np.newaxis]

    # Initialize the similarity matrix for ITEMS 
    similarityMatrix = np.zeros((userItemMatrix.shape[1], userItemMatrix.shape[1]))

    # Populate similarity between items in matrix n x n items 
    for i in range(similarityMatrix.shape[1]):
        for j in range(similarityMatrix.shape[1]):
            if i != j:
                # Users who rated both items
                commonUsers = np.intersect1d(np.nonzero(userItemMatrix[:, i]), np.nonzero(userItemMatrix[:, j]))

                # Compute the numerator: sum of products of adjusted ratings
                numerator = np.sum(adjustedMatrix[commonUsers, i] * adjustedMatrix[commonUsers, j])
                
                # Use precomputed denominators
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

def predictRating(userItemMatrix, similarityMatrix, unratedFile):
    # Predicting Ratings
    predictions = []
    array = []

    # Read the unrated data
    with open(unratedFile, 'r') as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            userID = int(row[0]) - 1
            itemID = int(row[1]) - 1
            timeStamp = int(row[2])

            # Get items that have been rated by user
            itemsRatedByUser = np.where(userItemMatrix[userID] > 0)[0]
            neighbourhood = itemsRatedByUser[similarityMatrix[itemID, itemsRatedByUser] > 0]

            array.append(len(neighbourhood))

            # Calculate the prediction of item-based recommender system using cosine similarity 
            numerator = np.sum(similarityMatrix[itemID, neighbourhood] * userItemMatrix[userID, neighbourhood])
            denominator = np.sum(similarityMatrix[itemID, neighbourhood])

            # Handle cold cases
            if denominator != 0:
                prediction = round(numerator / denominator)
                #prediction = numerator / denominator
            else:
                prediction = 1

            predictions.append([userID + 1, itemID + 1, prediction, timeStamp])

    
    np_array = np.array(array)

    # Calculate mean, max, and min (excluding zeros)
    mean_value = np.mean(np_array)
    max_value = np.max(np_array)

    # Filter out values equal to 0 and calculate min
    filtered_array = np_array[np_array > 0]
    if filtered_array.size > 0:
        min_value = np.min(filtered_array)
    else:
        min_value = None  # Or some other indication that there are no values > 0

    print("Mean:", mean_value)
    print("Max:", max_value)
    print("Min (excluding zeros):", min_value)
    
    return predictions

# Files
ratedFile = "train_100k_withratings.csv"
unratedFile = "test_100k_withoutratings.csv"
outputFile = "results3.csv"
#ratedFile = "test_30_rated.csv"
#unratedFile = "test_30_unrated.csv"
#outputFile = "test_results4.csv"


uiMatrix = userItemMatrix(ratedFile)
print(uiMatrix)
sMatrix = similarityMatrix(uiMatrix)
print(sMatrix)
predictions = predictRating(uiMatrix, sMatrix, unratedFile)

# Writing predictions to file
with open(outputFile, 'w', newline='') as file:
    csvWriter = csv.writer(file)
    for row in predictions:
        csvWriter.writerow(row)
