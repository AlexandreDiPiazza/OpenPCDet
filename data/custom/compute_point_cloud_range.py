import os
import numpy as np

# Specify the directory containing the .npy files
directory = './points'

# Initialize min and max values with None
min_values = [None, None, None]
max_values = [None, None, None]

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        # Load the numpy array from the .npy file
        array = np.load(os.path.join(directory, filename))
        
        # Update min and max values for each of the first three columns
        for i in range(3):
            column_min = np.min(array[:, i])
            column_max = np.max(array[:, i])
            
            if min_values[i] is None or column_min < min_values[i]:
                min_values[i] = column_min
            if max_values[i] is None or column_max > max_values[i]:
                max_values[i] = column_max

# Print the results
print(f"Minimum values for the first three columns (x,y,z): {min_values}")
print(f"Maximum values for the first three columns (x,y,z): {max_values}")