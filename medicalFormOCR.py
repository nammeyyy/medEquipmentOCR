import os
import re
import cv2
import pytesseract
from pytesseract import Output
from typhoon_ocr import ocr_document
from typing import Tuple

# กำหนดโฟลเดอร์ input และ output
input_folder = "./public/pre-processed"
output_folder = "./output/processed-results"

# สร้าง output folder ถ้ายังไม่มี
os.makedirs(output_folder, exist_ok=True)

def normalize_text(s: str) -> str:
    s = s.replace('\r', '\n')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n+', '\n', s).strip() 
    return s

def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost  # replace
            )
    return dp[la][lb]

def cer(ref: str, hyp: str) -> Tuple[float, int, int]:
    """
    Character Error Rate = edit_distance(chars) / len(ref_chars)
    return (cer_value, distance, ref_len)
    """
    ref_n = list(ref)
    hyp_n = list(hyp)
    d = edit_distance(ref_n, hyp_n)
    denom = max(1, len(ref_n))
    return d / denom, d, len(ref_n)

def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    """
    Word Error Rate = edit_distance(words) / len(ref_words)
    return (wer_value, distance, ref_len)
    """
    ref_w = ref.split()
    hyp_w = hyp.split()
    d = edit_distance(ref_w, hyp_w)
    denom = max(1, len(ref_w))
    return d / denom, d, len(ref_w)

def load_ref_label(gt_dir: str, image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(gt_dir, base + ".txt"),
        os.path.join(gt_dir, base + ".gt.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(
        f"ไม่พบ reference label สำหรับ {image_path} ใน {gt_dir} (คาดหวัง {base}.txt หรือ {base}.gt.txt)"
    )

def evaluate_one(ref_text: str, hyp_text: str) -> dict:
    ref_n = normalize_text(ref_text)
    hyp_n = normalize_text(hyp_text)
    cer_v, cer_ed, cer_den = cer(ref_n, hyp_n)
    wer_v, wer_ed, wer_den = wer(ref_n, hyp_n)
    return {
        "CER": cer_v, "CER_edits": cer_ed, "CER_ref_len": cer_den,
        "WER": wer_v, "WER_edits": wer_ed, "WER_ref_len": wer_den,
        "ref_chars": len(ref_n), "hyp_chars": len(hyp_n),
        "ref_words": len(ref_n.split()), "hyp_words": len(hyp_n.split()),
    }

gt_dir = "./reference"

# สถิติสะสม (แยก Typhoon/Tesseract)
agg = {
    "typhoon": {"cer_edits": 0, "cer_denom": 0, "wer_edits": 0, "wer_denom": 0, "count": 0},
    "tesseract": {"cer_edits": 0, "cer_denom": 0, "wer_edits": 0, "wer_denom": 0, "count": 0},
}

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, filename)
    print(f"\n=== Processing {filename} ===")

    # 1) โหลด Red Label
    try:
        gt_text = load_ref_label(gt_dir, img_path)
    except FileNotFoundError as e:
        print(f"SKIP (no GT): {e}")
        continue

    # 2) รัน Typhoon OCR
    try:
        typhoon_text = ocr_document(img_path)  # ได้สตริง markdown/ข้อความจาก Typhoon
    except Exception as e:
        print(f"Typhoon OCR error: {e}")
        typhoon_text = ""

    # 3) รัน Tesseract OCR
    try:
        image = cv2.imread(img_path)
        tess_text = pytesseract.image_to_string(image, lang="tha+eng")
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        tess_text = ""
    
    # Step 1: Typhoon OCR text
    markdown = ocr_document(img_path)

    # Step 2: Tesseract OCR bounding boxes
    image = cv2.imread(img_path)
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # confidence > 0
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #remark if it's too small

    # กำหนดชื่อไฟล์ output ให้ตรงกับชื่อ input
    base_name, _ = os.path.splitext(filename)

    # Save outputs
    bbox_path = os.path.join(output_folder, f"{base_name}_bbox.jpg")
    md_path = os.path.join(output_folder, f"{base_name}.md")

    cv2.imwrite(bbox_path, image)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved: {bbox_path}, {md_path}")

    # 4) ประเมินความแม่นยำ
    ty_eval = evaluate_one(gt_text, typhoon_text)
    te_eval = evaluate_one(gt_text, tess_text)

    # 5) แสดงผลต่อไฟล์
    def pct(x): return f"{x*100:.2f}%"
    print(f"[Typhoon] CER={pct(ty_eval['CER'])}  WER={pct(ty_eval['WER'])}")
    print(f"[Tesseract] CER={pct(te_eval['CER'])}  WER={pct(te_eval['WER'])}")

    # 6) สะสมสถิติเพื่อสรุปรวม (micro-average)
    agg["typhoon"]["cer_edits"] += ty_eval["CER_edits"]
    agg["typhoon"]["cer_denom"] += ty_eval["CER_ref_len"]
    agg["typhoon"]["wer_edits"] += ty_eval["WER_edits"]
    agg["typhoon"]["wer_denom"] += ty_eval["WER_ref_len"]
    agg["typhoon"]["count"] += 1

    agg["tesseract"]["cer_edits"] += te_eval["CER_edits"]
    agg["tesseract"]["cer_denom"] += te_eval["CER_ref_len"]
    agg["tesseract"]["wer_edits"] += te_eval["WER_edits"]
    agg["tesseract"]["wer_denom"] += te_eval["WER_ref_len"]
    agg["tesseract"]["count"] += 1

# 7) สรุปทั้งชุด
def summarize(name, a):
    cer_micro = a["cer_edits"] / max(1, a["cer_denom"])
    wer_micro = a["wer_edits"] / max(1, a["wer_denom"])
    print(f"\n=== SUMMARY: {name} ===")
    print(f"Files evaluated: {a['count']}")
    print(f"Micro-avg CER: {cer_micro*100:.2f}%")
    print(f"Micro-avg WER: {wer_micro*100:.2f}%")

summarize("Typhoon", agg["typhoon"])
summarize("Tesseract", agg["tesseract"])
