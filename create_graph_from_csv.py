import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- Configuration ---
CSV_FILE_PATH = './output/afterProcessed/ocr_eval.csv' # The path to your CSV file

# --- Matplotlib setup for Thai fonts (optional, but recommended) ---
# You may need to change 'Thonburi' to a font available on your Mac like 'Sarabun'
# For Windows, common fonts are 'Leelawadee' or 'Sarabun'
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Thonburi']
except Exception:
    print("Warning: Thai font not found. Labels may not render correctly.")
    pass

# --- Main Script Logic ---
try:
    # 1. Read the CSV file into a pandas DataFrame
    df = pd.read_csv(CSV_FILE_PATH)

    # 2. Verify required columns exist
    required_columns = ['category', 'CER', 'WER']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain the columns: {required_columns}")
        sys.exit(1)

    # 3. Group by category and calculate the average CER and WER
    # We drop rows where category might be empty
    df.dropna(subset=['category'], inplace=True)
    summary_df = df.groupby('category')[['CER', 'WER']].mean()

    # Prepare data for plotting
    categories = summary_df.index.tolist()
    cer_values = summary_df['CER'].tolist()
    wer_values = summary_df['WER'].tolist()

    print("--- Average Results from CSV ---")
    print(summary_df)
    print("--------------------------------")

    # 4. Create the grouped bar graph
    x = np.arange(len(categories))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, cer_values, width, label='Average CER (Character Error Rate)', color='skyblue')
    rects2 = ax.bar(x + width/2, wer_values, width, label='Average WER (Word Error Rate)', color='salmon')

    # Add labels, title, and ticks
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('OCR Performance Summary By Types Of Characters')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Function to attach a label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('ocr_summary_from_csv.png', dpi=300)
    plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at '{CSV_FILE_PATH}'")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
