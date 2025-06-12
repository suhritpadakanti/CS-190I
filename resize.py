import os
from PIL import Image

input_root = "output_images"
output_folder = "resized_images"
os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_root):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(root, filename)
            try:
                with Image.open(input_path) as img:
                    resized_img = img.resize((64, 64), Image.Resampling.LANCZOS)
                    subfolder = os.path.basename(root)
                    output_filename = f"{subfolder}_{filename}"
                    resized_img.save(os.path.join(output_folder, output_filename))
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

print("All images resized to 64x64 and saved to", output_folder)