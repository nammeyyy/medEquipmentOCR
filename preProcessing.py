import cv2
import os
import numpy as np

def preprocess_image_for_ocr(image_path, output_path):
    """
    Preprocess an image to improve OCR accuracy
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Remove noise with morphological operations
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Save the processed image
    cv2.imwrite(output_path, cleaned)
    return True

def process_folder(input_folder="./public", output_folder="./public/pre-processed"):
    """
    Process all images in the input folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Supported image extensions
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    
    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"processed_{filename}")
            
            print(f"Processing {filename}...")
            if preprocess_image_for_ocr(input_path, output_path):
                print(f"Successfully processed {filename}")
            else:
                print(f"Failed to process {filename}")

if __name__ == "__main__":
    process_folder()