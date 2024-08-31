import numpy as np

# Create a NumPy array
array = np.array([0, -1, 10, 20, 30, 40, 50, -5])

# Filter out values less than or equal to 0 and find the minimum
min_value = np.min(array[array > 0])

print("The minimum value greater than 0 is:", min_value)
