import os
import time
from rembg import remove
from PIL import Image
import io
import cv2
import numpy as np

# -------------------------------
# 🚀 CONFIG
# -------------------------------
input_folder = 'images/preprocessed_inputs'
output_folder = 'images/bg_removed_preprocess'

test_mode = False  # Set to False to process all images
test_files = ['2dtest.jpg']  # Only used if test_mode is True
show_preview = True  # Toggle image pop-up preview
post_process = True # Toggle post-processing (dilation/erosion)

# Get list of files to process
if test_mode:
    files_to_process = test_files
    print("🧪 Running in TEST MODE")
else:
    files_to_process = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    print(f"📁 Found {len(files_to_process)} images to process")

print(f"\n📂 Input folder: {input_folder}")
print(f"💾 Output folder: {output_folder}\n")

# Process each file
for filename in files_to_process:
    input_path = os.path.join(input_folder, filename)

    if not os.path.exists(input_path):
        print(f"⚠️  Skipping missing file: {filename}")
        continue

    output_filename = os.path.splitext(filename)[0] + '_bg_removed.png'
    output_path = os.path.join(output_folder, output_filename)

    print(f"🖼️  Processing: {filename}", end=' ... ')

    start_time = time.time()

    try:
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()

        output_data = remove(input_data)

        # Post-process: Dilation/Erosion to clean up edges (if enabled)
        if post_process:
            img = Image.open(io.BytesIO(output_data))
            img = img.convert("RGBA")
            data = np.array(img)

            # Extract alpha channel
            alpha = data[:, :, 3]

            # Morphological closing (dilate then erode)
            kernel = np.ones((7, 7), np.uint8)
            closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

            # Optional blur to smooth jagged edges
            smoothed = cv2.GaussianBlur(closed, (5, 5), 0)

            # Update the alpha channel
            data[:, :, 3] = smoothed

            # # Perform dilation on alpha channel (transparency channel)
            # kernel = np.ones((5, 5), np.uint8)
            # dilated = cv2.dilate(data[:, :, 3], kernel, iterations=1)

            # # Apply the modified alpha channel back
            # data[:, :, 3] = dilated

            # Update the image with post-processed data
            img = Image.fromarray(data)
            output_data = io.BytesIO()
            img.save(output_data, format="PNG")
            output_data = output_data.getvalue()

        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)

        if show_preview:
            img = Image.open(io.BytesIO(output_data))
            img.show()

        elapsed = time.time() - start_time
        print(f"✅ Saved to {output_filename} ({elapsed:.2f}s)")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

print("\n🎉 Done!")
