#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import sys
import unicodedata
from typing import List, Dict, Tuple, Set

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from typhoon_ocr import ocr_document as ty_ocr_document
except Exception:
    ty_ocr_document = None

import matplotlib.pyplot as plt

INPUT_DIR = "./public/pre-processed"
GT_DIR = "./reference"
OUT_DIR = "./output/afterProcessed"
LANG_TESS = "tha+eng"

THAI_DIGITS_MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Normalization & Cleaning ----------

def strip_markdown_layout(s: str) -> str:
    """
    Remove markdown-ish layout artifacts (tables, bullets, headers, code fences).
    """
    # Remove full lines that start with typical markdown/table markers
    s = re.sub(r'^\s*([#>\-\+\*`].*|\|.*)$', '', s, flags=re.MULTILINE)
    # Remove dangling table pipes inside lines
    s = s.replace('|', ' ')
    return s


def normalize_text_strict(s: str) -> str:
    """
    Normalize for fair text comparison:
    - Unicode NFC normalize (Thai diacritics compose)
    - Convert Thai digits to Arabic
    - Strip markdown/layout noise
    - Collapse whitespace and newlines
    """
    s = unicodedata.normalize('NFC', s)
    s = s.translate(THAI_DIGITS_MAP)
    s = strip_markdown_layout(s)
    s = s.replace('\r', '\n')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n+', '\n', s).strip()
    return s


def tokenize_chars(s: str) -> List[str]:
    return list(s)


def tokenize_words(s: str) -> List[str]:
    return s.split()


def edit_distance(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,    
                dp[i][j - 1] + 1,     
                dp[i - 1][j - 1] + cost  # replace
            )
    return dp[la][lb]


def cer(ref: str, hyp: str) -> Tuple[float, int, int]:
    a, b = tokenize_chars(ref), tokenize_chars(hyp)
    d = edit_distance(a, b)
    denom = max(1, len(a))
    return d / denom, d, denom


def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    a, b = tokenize_words(ref), tokenize_words(hyp)
    d = edit_distance(a, b)
    denom = max(1, len(a))
    return d / denom, d, denom


def cer_symmetric(ref: str, hyp: str) -> float:
    a, b = tokenize_chars(ref), tokenize_chars(hyp)
    d = edit_distance(a, b)
    denom = max(1, (len(a) + len(b)) / 2.0)
    return d / denom


def similarity_ratio(ref: str, hyp: str) -> float:
    """
    Levenshtein similarity normalized by max length (0..1), 1.0 is perfect.
    """
    a, b = tokenize_chars(ref), tokenize_chars(hyp)
    d = edit_distance(a, b)
    denom = max(1, max(len(a), len(b)))
    return 1.0 - (d / denom)

def normalize_item(s: str) -> str:
    s = normalize_text_strict(s)
    s = s.lower()
    # Remove trailing quantities and symbols
    s = re.sub(r'[\s\*]+$', '', s)
    return s.strip()


ITEM_LINE_PATTERNS = [
    # "1. Item name    2"
    re.compile(r'^\s*(\d+)[\.\)]\s*(.+?)\s+(\d+)\s*$', re.IGNORECASE),
    # "Item name    2" (no index)
    re.compile(r'^\s*([^\d\W].*?)\s+(\d+)\s*$', re.IGNORECASE),
    # "1 Item name"
    re.compile(r'^\s*(\d+)\s+(.+?)\s*$', re.IGNORECASE),
]


def extract_items(text: str) -> Tuple[Set[str], int]:
    """
    Heuristic: extract set of item names (normalized) and count of lines matched.
    """
    items = set()
    matched = 0
    for line in normalize_text_strict(text).split('\n'):
        line = line.strip()
        if not line:
            continue
        ok = False
        for pat in ITEM_LINE_PATTERNS:
            m = pat.match(line)
            if m:
                # choose the group that looks like item name (last textual group)
                groups = m.groups()
                # pick the longest alpha-ish group as item
                candidates = [g for g in groups if g and re.search(r'[A-Za-zก-ฮ]', g)]
                if candidates:
                    item = normalize_item(max(candidates, key=len))
                    if item:
                        items.add(item)
                        matched += 1
                        ok = True
                        break
        if not ok:
            # fallback: lines with many letters and no colon look like names
            if re.search(r'[A-Za-zก-ฮ].{2,}', line) and ':' not in line:
                items.add(normalize_item(line))
    return items, matched


def item_metrics(ref_text: str, hyp_text: str) -> Dict[str, float]:
    ref_items, _ = extract_items(ref_text)
    hyp_items, _ = extract_items(hyp_text)
    inter = ref_items.intersection(hyp_items)
    p = (len(inter) / max(1, len(hyp_items))) if hyp_items else 0.0
    r = (len(inter) / max(1, len(ref_items))) if ref_items else 0.0
    f1 = (2 * p * r / max(1e-12, (p + r))) if (p + r) > 0 else 0.0
    return {
        "items_ref": len(ref_items),
        "items_hyp": len(hyp_items),
        "items_match": len(inter),
        "items_precision": p,
        "items_recall": r,
        "items_f1": f1,
    }

def load_reference_label(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(GT_DIR, base + ".txt"),
        os.path.join(GT_DIR, base + ".gt.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"Reference label not found for {image_path}; expected {candidates}")


def run_typhoon(img_path: str) -> str:
    if ty_ocr_document is None:
        raise RuntimeError("Typhoon OCR not available")
    return str(ty_ocr_document(img_path))


def run_tesseract(img_path: str) -> str:
    if pytesseract is None or cv2 is None:
        raise RuntimeError("Tesseract or OpenCV not available")
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    return pytesseract.image_to_string(image, lang=LANG_TESS)

def evaluate_one(ref_text: str, hyp_text: str) -> Dict[str, float]:
    ref_n = normalize_text_strict(ref_text)
    hyp_n = normalize_text_strict(hyp_text)
    cer_v, cer_ed, cer_den = cer(ref_n, hyp_n)
    wer_v, wer_ed, wer_den = wer(ref_n, hyp_n)
    sim_v = similarity_ratio(ref_n, hyp_n)
    cer_sym_v = cer_symmetric(ref_n, hyp_n)
    item = item_metrics(ref_n, hyp_n)
    result = {
        "CER": cer_v, "CER_edits": cer_ed, "CER_ref_len": cer_den,
        "WER": wer_v, "WER_edits": wer_ed, "WER_ref_len": wer_den,
        "SIM": sim_v, "CER_sym": cer_sym_v,
        "ref_chars": len(ref_n), "hyp_chars": len(hyp_n),
        "ref_words": len(ref_n.split()), "hyp_words": len(hyp_n.split()),
    }
    result.update(item)
    return result


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def main():
    ensure_dirs()
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images.sort()
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return 1

    engines = []
    if ty_ocr_document is not None:
        engines.append("typhoon")
    if pytesseract is not None and cv2 is not None:
        engines.append("tesseract")
    if not engines:
        print("No OCR engines available. Please install typhoon_ocr / opencv-python / pytesseract.")
        return 1

    rows: List[Dict] = []
    # For micro-averages
    agg = {e: {"cer_edits": 0, "cer_denom": 0, "wer_edits": 0, "wer_denom": 0, "count": 0} for e in engines}

    for img in images:
        img_path = os.path.join(INPUT_DIR, img)
        try:
            gt = load_reference_label(img_path)
        except FileNotFoundError as e:
            print(f"SKIP (no GT): {e}")
            continue

        for engine in engines:
            try:
                hyp = run_typhoon(img_path) if engine == "typhoon" else run_tesseract(img_path)
            except Exception as e:
                print(f"{engine} error on {img}: {e}")
                hyp = ""

            m = evaluate_one(gt, hyp)
            row = {
                "image": img,
                "engine": engine,
                "CER": m["CER"],
                "WER": m["WER"],
                "CER_sym": m["CER_sym"],
                "SIM": m["SIM"],
                "items_precision": m["items_precision"],
                "items_recall": m["items_recall"],
                "items_f1": m["items_f1"],
                "items_ref": m["items_ref"],
                "items_hyp": m["items_hyp"],
                "items_match": m["items_match"],
            }
            rows.append(row)

            # micro-avg accumulators
            agg[engine]["cer_edits"] += m["CER_edits"]
            agg[engine]["cer_denom"] += m["CER_ref_len"]
            agg[engine]["wer_edits"] += m["WER_edits"]
            agg[engine]["wer_denom"] += m["WER_ref_len"]
            agg[engine]["count"] += 1

            print(f"{img} [{engine}] -> CER={pct(m['CER'])}, WER={pct(m['WER'])}, SIM={m['SIM']:.3f}, F1={m['items_f1']:.3f}")

    # Save CSV
    if rows:
        csv_path = os.path.join(OUT_DIR, "ocr_eval.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved CSV: {csv_path}")
    else:
        print("No evaluation rows produced (missing GT or OCR errors).")
        return 0

    # Plots: CER, WER, SIM
    rows_sorted = sorted(rows, key=lambda r: (r["image"], r["engine"]))
    labels = [f"{r['image']}\n{r['engine']}" for r in rows_sorted]

    def bar_plot(metric: str, ylabel: str, title: str, filename: str):
        vals = [r[metric] for r in rows_sorted]
        if not vals:
            return
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, filename)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")

    bar_plot("CER", "CER", "Character Error Rate (lower is better)", "cer_plot.png")
    bar_plot("WER", "WER", "Word Error Rate (lower is better)", "wer_plot.png")
    bar_plot("SIM", "Similarity", "Levenshtein Similarity (higher is better)", "sim_plot.png")

    # Print summaries
    for e, a in agg.items():
        if a["count"] == 0:
            continue
        cer_micro = a["cer_edits"] / max(1, a["cer_denom"])
        wer_micro = a["wer_edits"] / max(1, a["wer_denom"])
        print(f"\n=== SUMMARY: {e} ===")
        print(f"Files evaluated: {a['count']}")
        print(f"Micro-avg CER: {cer_micro*100:.2f}%")
        print(f"Micro-avg WER: {wer_micro*100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())