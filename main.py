import streamlit as st
import pandas as pd
import sqlite3
import tempfile
import os
import pdfplumber
from PIL import Image
from datetime import datetime
import re
from openai import OpenAI
from dotenv import load_dotenv
import json
import fitz  # PyMuPDF
import base64
from io import BytesIO
import time

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup DB
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect("database/vendor_gl.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS vendor_gl (
        vendor TEXT,
        corrected_vendor TEXT,
        gl_account TEXT,
        last_used TEXT,
        usage_count INTEGER
    )
''')
conn.commit()

# Load COA
COA_PATH = "data/FirmCOAv1.xlsx"
if not os.path.exists(COA_PATH):
    st.error("Chart of Accounts file is missing.")
    st.stop()
coa_df = pd.read_excel(COA_PATH)
coa_df.columns = [col.strip() for col in coa_df.columns]
coa_df = coa_df[['GL Account', 'Account Name', 'Sample Vendors']].dropna()

def gpt_with_retry(client, **kwargs):
    for _ in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            print("[GPT retry] Error:", e)
            time.sleep(2)
    return None

def clean_vendor(desc):
    desc = re.sub(r"^\d{2}/\d{2}\s+", "", desc)
    return re.sub(r"\b(card|transaction|\d{4,})\b", "", desc, flags=re.IGNORECASE).strip()

def match_category(vendor, coa_df, cursor):
    vendor_cleaned = vendor.lower().strip()
    try:
        cursor.execute("SELECT gl_account FROM vendor_gl WHERE corrected_vendor = ?", (vendor_cleaned,))
        result = cursor.fetchone()
        if result:
            cursor.execute("UPDATE vendor_gl SET usage_count = usage_count + 1, last_used = ? WHERE corrected_vendor = ?",
                           (datetime.now().strftime('%Y-%m-%d'), vendor_cleaned))
            conn.commit()
            return result[0]
    except Exception as e:
        print("[DB] Read error:", e)

    prompt = f"""
    You are an accountant. Pick the best GL Account for vendor: {vendor}.
    Chart:
    {coa_df[['GL Account', 'Account Name']].to_string(index=False)}
    Reply JSON: {{"gl_account": "...", "confidence": 0.95}}
    """
    try:
        response = gpt_with_retry(client,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful accountant."},
                {"role": "user", "content": prompt}
            ]
        )
        if response:
            result = json.loads(response.choices[0].message.content.strip())
            gl = result.get("gl_account")
            conf = result.get("confidence", 0)
            if conf >= 0.9:
                cursor.execute("INSERT INTO vendor_gl VALUES (?, ?, ?, ?, ?)",
                               (vendor_cleaned, vendor, gl, datetime.now().strftime('%Y-%m-%d'), 1))
                conn.commit()
                return gl
    except Exception as e:
        print("[GPT Match Error]:", e)

    for _, row in coa_df.iterrows():
        if str(row["Sample Vendors"]).lower() in vendor_cleaned:
            return row["GL Account"]

    return None

def ai_ocr_from_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    try:
        response = gpt_with_retry(client,
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "Extract transactions from image."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract transactions"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        if response:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print("[OCR Vision]:", e)
    return ""

def extract_text_from_image(image):
    try:
        import pytesseract
        return pytesseract.image_to_string(image)
    except:
        return ai_ocr_from_image(image)

st.set_page_config(page_title="Bank Statement Categorizer", layout="centered")

with st.container():
    st.image("data/ValiantIconWhite.png", width=250)
    st.title("Bank Statement Categorizer")
    st.markdown("Upload PDF statements to categorize transactions.")

uploaded_file = st.file_uploader("📄 Upload Bank Statement (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "temp.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        transactions = []
        progress_bar = st.progress(0)

        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            progress_bar.progress((i + 1) / len(doc))
            pix = page.get_pixmap(dpi=300)
            img_path = os.path.join(tmpdir, f"page_{i}.png")
            pix.save(img_path)
            image = Image.open(img_path)

            text = extract_text_from_image(image)
            if not text or len(text.strip()) < 20:
                try:
                    with pdfplumber.open(file_path) as fallback_pdf:
                        text = fallback_pdf.pages[i].extract_text()
                except:
                    text = ""

            if not text or "intentionally left blank" in text.lower():
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for j, line in enumerate(lines):
                date_match = re.match(r'(\\d{2}/\\d{2})\\s+', line)
                if date_match:
                    full_line = line
                    if j + 1 < len(lines):
                        full_line += " " + lines[j + 1]
                    match = re.match(r'(\\d{2}/\\d{2})[^\\d]*(.*?)\\s+([-+]?\\$?\\d+[,.]\\d{2})', full_line)
                    if match:
                        date_raw, description, amount_str = match.groups()
                        try:
                            parsed_date = datetime.strptime(date_raw + "/2025", "%m/%d/%Y")
                        except:
                            continue
                        amount = float(amount_str.replace("$", "").replace(",", ""))
                        vendor = clean_vendor(description)
                        category = match_category(vendor, coa_df, cursor)
                        transactions.append({
                            "Date": parsed_date.strftime("%Y-%m-%d"),
                            "Description": description,
                            "Amount": amount,
                            "Vendor": vendor,
                            "Category": category,
                            "Status": "Auto-Categorized" if category else "Uncategorized",
                            "Reason": "Matched" if category else "Needs Review"
                        })

        progress_bar.empty()
        df = pd.DataFrame(transactions)

        # 🧠 Category dropdown support
        category_options = sorted(coa_df["GL Account"].astype(str) + " " + coa_df["Account Name"])
        df["Category"] = df["Category"].astype(str)

        edited_df = st.data_editor(
            df[["Date", "Description", "Amount", "Vendor", "Category", "Status", "Reason"]],
            column_config={
                "Category": st.column_config.SelectboxColumn("Category", options=category_options, required=False)
            },
            num_rows="dynamic",
            use_container_width=True
        )

        for _, row in edited_df.iterrows():
            if row["Vendor"] and row["Category"]:
                vendor = row["Vendor"].strip()
                vendor_cleaned = vendor.lower()
                gl = row["Category"]

                cursor.execute("SELECT usage_count FROM vendor_gl WHERE corrected_vendor = ?", (vendor_cleaned,))
                result = cursor.fetchone()
                usage_count = (result[0] if result else 0) + 1

                cursor.execute("""
                    INSERT OR REPLACE INTO vendor_gl (
                        vendor, corrected_vendor, gl_account, last_used, usage_count
                    ) VALUES (?, ?, ?, ?, ?)
                """, (vendor_cleaned, vendor, gl, datetime.now().strftime('%Y-%m-%d'), usage_count))
                conn.commit()

        st.download_button("Download CSV", data=edited_df.to_csv(index=False), file_name="categorized_transactions.csv", mime="text/csv")
