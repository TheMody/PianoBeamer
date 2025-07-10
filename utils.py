import os
from PIL import Image 


path = "data/images/"

filename = 0
for file in os.listdir(path):
    if file.endswith('.jpg') or file.endswith('.png'):
        image_path = os.path.join(path, file)
        try:
            filename += 1
            with Image.open(image_path) as img:
                print(f"Image {file} opened successfully.")
                #save image with new name
                new_filename = f"image_{filename}.png"
                img.save(os.path.join(path, new_filename))
        except Exception as e:
            print(f"Error opening image {file}: {e}")
    else:
        print(f"Skipping non-image file: {file}")