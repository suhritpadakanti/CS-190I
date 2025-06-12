import os
import shutil

input_dir = "augmented_images"
dummy_class_dir = os.path.join(input_dir, "images")

# Make the dummy class directory
os.makedirs(dummy_class_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    fpath = os.path.join(input_dir, fname)
    if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        shutil.move(fpath, os.path.join(dummy_class_dir, fname))

print(f"Moved images into: {dummy_class_dir}")
