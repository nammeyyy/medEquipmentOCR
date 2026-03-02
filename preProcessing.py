import cv2
import os
import numpy as np

def extract_columns(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = cv2.imread(image_path)
    if img is None: return 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ใช้ BINARY_INV เพื่อเน้นเส้นตาราง
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # เน้นหาเส้นตั้งยาวๆ เพื่อแยกคอลัมน์ (ปรับค่า 100 ตามความละเอียดรูป)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    # หาตำแหน่งเส้นตั้งเพื่อหาขอบซ้าย-ขวาของแต่ละคอลัมน์
    contours, _ = cv2.findContours(detect_vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # ดึงค่าพิกัด X ของเส้นตั้งออกมาและเรียงลำดับจากซ้ายไปขวา
    boundaries = sorted([cv2.boundingRect(c)[0] for c in contours])
    
    # กำจัดเส้นที่อยู่ใกล้กันเกินไป (เส้นซ้อน)
    filtered_boundaries = []
    if boundaries:
        filtered_boundaries.append(boundaries[0])
        for i in range(1, len(boundaries)):
            # ถ้าห่างกันเกิน 50px ถึงนับเป็นคอลัมน์ใหม่ (ปรับเลขนี้ได้ถ้าคอลัมน์แคบไปหรือกว้างไป)
            if boundaries[i] - filtered_boundaries[-1] > 50: 
                filtered_boundaries.append(boundaries[i])

    # ตัดรูปตามคอลัมน์
    count = 0
    for i in range(len(filtered_boundaries) - 1):
        x_start = filtered_boundaries[i]
        x_end = filtered_boundaries[i+1]
        column_img = img[:, x_start:x_end] # ตัดยาวตั้งแต่บนสุดถึงล่างสุด
        cv2.imwrite(os.path.join(output_folder, f"column_{i}.jpg"), column_img)
        count += 1
    
    return count

def preprocess_image_for_ocr(image_path, output_path):
    """
    ทำให้ภาพชัดขึ้นก่อนส่งให้ OCR (ทำขาวดำ)
    """
    img = cv2.imread(image_path)
    if img is None: return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(output_path, thresh)
    return True

def process_folder(input_folder="./public", output_folder="./public/pre-processed", cells_base_folder="./public/cells"):
    """
    ลูปประมวลผลทุกรูปในโฟลเดอร์
    """
    if not os.makedirs(output_folder, exist_ok=True): pass
    
    supported_extensions = ('.png', '.jpg', '.jpeg')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            
            # 1. ทำภาพขาวดำ
            processed_path = os.path.join(output_folder, f"processed_{filename}")
            print(f"--- Processing {filename} ---")
            preprocess_image_for_ocr(input_path, processed_path)
            
            # 2. ตัดรูปเป็นคอลัมน์ (เรียกใช้ฟังก์ชัน extract_columns ให้ตรงกัน)
            image_name = os.path.splitext(filename)[0]
            current_cells_folder = os.path.join(cells_base_folder, image_name)
            
            num_cols = extract_columns(input_path, current_cells_folder)
            print(f"Success: Extracted {num_cols} columns into {current_cells_folder}")

if __name__ == "__main__":
    process_folder()

# def preprocess_image_for_ocr(image_path, output_path):
#     """
#     Preprocess an image to improve OCR accuracy
#     """
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not read image {image_path}")
#         return False
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
    
#     # Remove noise with morphological operations
#     kernel = np.ones((1, 1), np.uint8)
#     cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
#     # Save the processed image
#     cv2.imwrite(output_path, cleaned)
#     return True

# def process_folder(input_folder="./public", output_folder="./public/pre-processed"):
#     """
#     Process all images in the input folder
#     """
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # Supported image extensions
#     supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    
#     # Process each image in the folder
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(supported_extensions):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, f"processed_{filename}")
            
#             print(f"Processing {filename}...")
#             if preprocess_image_for_ocr(input_path, output_path):
#                 print(f"Successfully processed {filename}")
#             else:
#                 print(f"Failed to process {filename}")

# if __name__ == "__main__":
#     process_folder()