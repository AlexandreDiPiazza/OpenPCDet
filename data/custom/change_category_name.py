import os

def rename_category(folder_path):
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Read the file's content
            with open(file_path, 'r') as file:
                content = file.readlines()
            
            # Replace "Adult" with "Pedestrian" in the content
            modified_content = [line.replace('Pedestrian', 'Adult') for line in content]
            
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(modified_content)

# Example usage
folder_path = './labels'
rename_category(folder_path)