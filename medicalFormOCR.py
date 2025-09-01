import os
import cv2
import pytesseract
from pytesseract import Output
from typhoon_ocr import ocr_document

# กำหนดโฟลเดอร์ input และ output
input_folder = "./public"
output_folder = "./output"

# สร้าง output folder ถ้ายังไม่มี
os.makedirs(output_folder, exist_ok=True)

# loop ทุกไฟล์ใน public/
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)

        print(f"Processing {filename}...")

        # Step 1: Typhoon OCR text
        markdown = ocr_document(img_path)

        # Step 2: Tesseract OCR bounding boxes
        image = cv2.imread(img_path)
        data = pytesseract.image_to_data(image, output_type=Output.DICT)

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # confidence > 0
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # กำหนดชื่อไฟล์ output ให้ตรงกับชื่อ input
        base_name, _ = os.path.splitext(filename)

        # Save outputs
        bbox_path = os.path.join(output_folder, f"{base_name}_bbox.jpg")
        md_path = os.path.join(output_folder, f"{base_name}.md")

        cv2.imwrite(bbox_path, image)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Saved: {bbox_path}, {md_path}")