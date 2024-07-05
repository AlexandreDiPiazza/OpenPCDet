import os
import random

# Set a seed for reproducibility
random.seed(42)

# Specify the directory containing the txt files
directory = './labels'

# List to hold the names of the txt files without the extension
file_names_without_extension = []

# Set to hold unique categories
unique_categories = set()

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # Remove the .txt extension and add the name to the list
        file_names_without_extension.append(os.path.splitext(filename)[0])
        
        # Construct full file path
        file_path = os.path.join(directory, filename)
        # Open and read the file
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into components and extract the last one
                category = line.strip().split()[-1]
                # Add the category to the set
                unique_categories.add(category)

# Optionally, print or save the sorted list of unique categories
print('All the categories: ', unique_categories)