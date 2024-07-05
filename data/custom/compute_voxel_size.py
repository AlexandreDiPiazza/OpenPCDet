import numpy as np

pc_path = "/home/ubuntu/alex/OpenPCDet/data/custom/points/000000491305.npy"
# Load the point cloud array from a file
point_cloud = np.load(pc_path)

# Sort the point cloud based on the x-coordinate (first column)
point_cloud_sorted = point_cloud[np.argsort(point_cloud[:, 0])]

# Initialize a list to store average voxel sizes for x, y, z dimensions
average_voxel_sizes = []

# For each spatial dimension (x, y, z)
for i in range(3):
    # Calculate differences between consecutive points in the sorted array
    differences = np.diff(point_cloud_sorted[:, i])
    
    # Compute the average difference (voxel size) for the current dimension
    average_voxel_size = np.mean(differences)
    average_voxel_sizes.append(average_voxel_size)

# Print the average voxel size for each dimension
print(f"Average voxel sizes (x, y, z): {average_voxel_sizes}")