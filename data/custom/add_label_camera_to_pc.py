import numpy as np
import os


def add_dimension_and_save(input_folder, output_folder, new_value):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    for file_name in os.listdir(input_folder):
        # Check if the file is a .npy file
        if file_name.endswith(".npy"):
            # Construct the full file path
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Load the numpy array from the input file
            data = np.load(input_file_path)
            print('before:', np.shape(data), data[0:3, :])
            # Check if the data is already in the desired shape
            if data.shape[1] != 4:
                print(f"Skipping {file_name}: unexpected shape {data.shape}")
                continue

            # Add a new column with the specified new value
            new_column = np.full((data.shape[0], 1), new_value)
            new_data = np.hstack((data, new_column))
            print('after:', np.shape(new_data), new_data[0:3, :])
            # Save the modified array to the output file
            np.save(output_file_path, new_data)
            print(f"Processed and saved: {file_name}")


# Example usage
input_folder = "/home/ubuntu/alex/OpenPCDet/data/custom/full_data/Batch_1_July_5th/2023-10-16-10-06-41_3_1697420355733614_updated/lidar_4/points_openpcdet_format"
output_folder = "/home/ubuntu/alex/OpenPCDet/data/custom/full_data/Batch_1_July_5th/2023-10-16-10-06-41_3_1697420355733614_updated/lidar_4/points_openpcdet_format_camera_label"
new_value = 3  # Example new value to add
add_dimension_and_save(input_folder, output_folder, new_value)
