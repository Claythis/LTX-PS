from skimage import io, img_as_float, restoration, exposure, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import os
import time

"""
Removing unwanted noise from background and preserving important details

Non-Local Means (NLM) - provides better detail retention

"""

image = img_as_float(io.imread('images/inputs/pixar_image.jpg'))

""" 
Removing unwanted noise from background and preserving important details
Non-Local Means (NLM) - provides better detail retention
"""

def denoise_image(image): 
    denoised_image = restoration.denoise_nl_means(
        image,
        h = 0.1, #Smoothing parameter between 0 and 1 
        fast_mode=True,
        #Increase to improve denoising process but increase computation time
        patch_size = 5,
        patch_distance = 8,
        channel_axis = -1
    )
    return denoised_image


def color_correction_and_normalization(image): 
    #Apply Contrast Enhancement (CLAHE)
    clahe_img = exposure.equalize_adapthist(image, clip_limit=0.03)
    #Normalize Intensity Range
    rescaled_img = exposure.rescale_intensity(clahe_img, in_range='image', out_range=(0, 1))
    gamma_img = exposure.adjust_gamma(rescaled_img, gamma=0.8)
    return gamma_img

def crop_image(image): 
    return None

def resize_image(image, size):
    """ 
    Resize image to size(width, height)
    # resized = resize_image_skimage(image, (256, 256, 3))  # For RGB images
    """
    resized = resize(image, size, anti_aliasing=True, preserve_range=True)
    return resized.astype(image.dtype)

def plot_image(final_img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(final_img)
    axes[1].set_title('Enhanced & Normalized')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

input_folder = 'images/inputs'
output_folder = 'images/preprocessed_inputs'

def save_process(): 
    #Get list of files to preprocess
    files_to_process = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    #preprocess each file
    for filename in files_to_process:
        input_path = os.path.join(input_folder, filename)

        if not os.path.exists(input_path):
            print(f"Skipping missing file: {filename}")
            continue

        output_filename = os.path.splitext(filename)[0] + '_preprocessed.png'
        output_path = os.path.join(output_folder, output_filename)

        start_time = time.time()

        image = img_as_float(io.imread(input_path))

        # Apply preprocessing steps
        denoised = denoise_image(image)
        normalized = color_correction_and_normalization(denoised)

        #Correct Datatyle
        normalized_uint8 = img_as_ubyte(normalized)
        io.imsave(output_path, normalized_uint8)

        end_time = time.time()
        print(f"Processed {filename} in {end_time - start_time:.2f} seconds → Saved to {output_filename}")

if __name__ == "__main__":
    save_process()