#!/usr/bin/env python3
import sys
import json
import argparse
import logging
import re
import pdfplumber
import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def remove_duplicate_table_text(full_text):
    pattern = r"3/7::.*?LNG Side"
    cleaned = re.sub(pattern, "", full_text, flags=re.DOTALL)
    return cleaned

def extract_table_text(pdf_path):
    table_data = []
    page_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]

        raw_table = page.extract_table()
        if not raw_table:
            logging.warning("No table found on Page 1.")
            return [], ""

        header = raw_table[0]
        data_rows = raw_table[1:]
        for i, row in enumerate(data_rows[:10]):
            while len(row) < 5:
                row.append("")
            table_row = {
                "Category": row[0].strip(),
                "Direction": "",  # We'll fill this later with OpenCV detection
                "Current": row[2].strip(),
                "Outlook": row[3].strip(),
                "Notes": row[4].strip()
            }
            table_data.append(table_row)

        full_text = page.extract_text() or ""
        no_duplicate = remove_duplicate_table_text(full_text)
        page_text = " ".join(no_duplicate.split())

    return table_data, page_text

def extract_non_table_text(pdf_path, table_pages=[0, 3]):
    """
    Extracts and cleans text from PDF pages that do not contain tables.

    Args:
        pdf_path (str): The path to the PDF file.
        table_pages (list, optional): List of page indices that contain tables.
                                      Defaults to [0, 3].

    Returns:
        str: A single string containing the cleaned text from non-table pages.
    """
    non_table_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        # Iterate over all pages in the PDF
        for i, page in enumerate(pdf.pages):
            # Process only pages that are not in the table_pages list
            if i not in table_pages:
                page_text = page.extract_text() or ""
                non_table_text += " " + page_text

    # Remove extra whitespace and return the cleaned text
    cleaned_text = " ".join(non_table_text.split())
    return cleaned_text

def render_pdf_page_to_png(pdf_path, page_num=0, dpi=150, out_png="page.png"):
    pages = convert_from_path(pdf_path, dpi=dpi)
    if page_num >= len(pages):
        raise ValueError(f"PDF only has {len(pages)} pages, can't extract page {page_num}")
    pages[page_num].save(out_png, "PNG")
    logging.info(f"Rendered page {page_num} to {out_png} at {dpi} dpi")

def find_circles_in_image(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read {image_path}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Adjust these HSV ranges to match your PDF circles
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 80, 80])
    upper_green = np.array([80, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    circles_red = find_circle_contours(mask_red, "Red")
    circles_green = find_circle_contours(mask_green, "Green")
    return circles_red + circles_green  # each is (cx, cy, color)

def find_circle_contours(mask, color_label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / (h + 1e-5)
        if 0.7 < ratio < 1.3:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            found.append((cx, cy, color_label))
    return found

def sort_circles_top_to_bottom(circles):
    return sorted(circles, key=lambda c: c[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--output", default="page1_extracted.json", help="Base name for output JSON")
    args = parser.parse_args()

    pdf_path = args.pdf
    base_output = args.output

    # 1) Extract table data from pages with tables (pages 0 and 3; currently, table extraction is from page 0)
    table_data, page_text = extract_table_text(pdf_path)

    # 2) Extract text from non-table pages (pages 1, 2, 4, and 5)
    non_table_text = extract_non_table_text(pdf_path, table_pages=[0, 3])

    data = {
        "table_data": table_data,
        "NaturalGasFundamentalAnalysisPG1": page_text,
        "NonTableText": non_table_text
    }

    # 3) Render PDF page 0 to PNG (for circle detection on table page)
    out_png = "page.png"
    render_pdf_page_to_png(pdf_path, page_num=0, dpi=150, out_png=out_png)

    # 4) Detect circles
    circles = find_circles_in_image(out_png)
    if not circles:
        logging.warning("No circles found; 'Direction' fields remain blank.")
    else:
        circles_sorted = sort_circles_top_to_bottom(circles)
        min_len = min(len(data["table_data"]), len(circles_sorted))
        for i in range(min_len):
            color_label = circles_sorted[i][2]
            data["table_data"][i]["Direction"] = color_label

    # 5) Generate a timestamped filename
    now_str = datetime.now().strftime("%d_%B_%H%M")  # e.g. "23_March_2307"
    filename_no_ext, ext = os.path.splitext(base_output)
    output_filename = f"{filename_no_ext}_{now_str}{ext}"

    # 6) Save final JSON with the new timestamped filename
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logging.info(f"Saved combined table, circle colors, and non-table text to {output_filename}")

if __name__ == "__main__":
    main()
