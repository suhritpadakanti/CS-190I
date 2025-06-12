import os

folder_path = "./augmented_images"

file_count = len([
    f for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
])

print(f"Number of files: {file_count}")