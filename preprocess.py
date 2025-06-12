import os
from PIL import Image

# --- Configuration ---
INPUT_FOLDER = "raw_images"  # Folder with your subdirectories like 'Apple', 'Facebook', etc.
OUTPUT_FOLDER = "data/emojis" # Output folder for ALL resized images
IMAGE_SIZE = (64, 64)

def preprocess_images_with_subfolders():
    """
    Finds images in subdirectories of the input folder, resizes them,
    and saves them to a single output folder.
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        print("Please ensure your 'raw_images' folder exists.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Searching for images in '{INPUT_FOLDER}' and its subfolders...")
    images_processed = 0
    
    # os.walk will go through the root folder and all its subdirectories
    for root, _, files in os.walk(INPUT_FOLDER):
        for filename in files:
            # Check for valid image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_path = os.path.join(root, filename)
                
                # Create a unique name to prevent files from different folders
                # with the same name from overwriting each other.
                # Example: 'Apple_image1.png', 'Google_image1.png'
                subfolder_name = os.path.basename(root)
                unique_filename = f"{subfolder_name}_{filename}"
                output_path = os.path.join(OUTPUT_FOLDER, unique_filename)

                try:
                    with Image.open(input_path) as img:
                        # Convert to RGB to handle images with transparency (like PNGs)
                        resized_img = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        resized_img.save(output_path)
                        images_processed += 1
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")
    
    if images_processed > 0:
        print(f"\n✅ Success! Processed {images_processed} images.")
        print(f"Resized images are ready in '{OUTPUT_FOLDER}'. You can now run train.py.")
    else:
        print("\n❌ Warning: No valid image files were found to process.")


if __name__ == "__main__":
    preprocess_images_with_subfolders()