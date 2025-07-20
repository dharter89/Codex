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
import random

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

# Initialize SQLite connection and cursor
conn = sqlite3.connect("database/vendor_gl.db", check_same_thread=False)
cursor = conn.cursor()

# Recreate vendor_gl table if corrupted or missing
try:
    cursor.execute("SELECT 1 FROM vendor_gl LIMIT 1")
except sqlite3.OperationalError:
    cursor.execute("DROP TABLE IF EXISTS vendor_gl")
    cursor.execute('''
        CREATE TABLE vendor_gl (
            vendor TEXT,
            corrected_vendor TEXT,
            gl_account TEXT,
            last_used TEXT,
            usage_count INTEGER
        )
    ''')
    conn.commit()

# Load Chart of Accounts
COA_PATH = "data/FirmCOAv1.xlsx"
if not os.path.exists(COA_PATH):
    st.error("Chart of Accounts file is missing. Please place 'FirmCOAv1.xlsx' in the 'data' folder.")
    st.stop()
coa_df = pd.read_excel(COA_PATH)
coa_df.dropna(how='all', inplace=True)
coa_df.columns = [col.strip() for col in coa_df.columns]
coa_df = coa_df[['GL Account', 'Account Name', 'Sample Vendors']].dropna(subset=['GL Account', 'Sample Vendors'])

def gpt_with_retry(client, **kwargs):
    max_retries = 5
    backoff = 1
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
            else:
                print("GPT call failed:", e)
                break
    return None

def clean_vendor(description):
    description = re.sub(r"^\d{2}/\d{2}\s+", "", description)
    vendor = re.sub(r"\b(card|transaction|\d{4,})\b", "", description, flags=re.IGNORECASE)
    return vendor.strip()

def match_category(vendor, coa_df, cursor):
    vendor_cleaned = str(vendor or "").strip().lower()

    try:
        cursor.execute("SELECT gl_account FROM vendor_gl WHERE corrected_vendor = ?", (vendor_cleaned,))
        result = cursor.fetchone()
        if result:
            cursor.execute("UPDATE vendor_gl SET usage_count = usage_count + 1, last_used = ? WHERE corrected_vendor = ?",
                           (datetime.now().strftime('%Y-%m-%d'), vendor_cleaned))
            conn.commit()
            return result[0]
    except Exception as e:
        print("[ERROR] DB read failed:", e)

    try:
        prompt = f"""
        You are an accountant. Based on the chart of accounts, pick the best GL Account for vendor: {vendor}.
        Chart:
        {coa_df[['GL Account', 'Account Name']].to_string(index=False)}
        Reply with JSON: {{"gl_account": "...", "confidence": 0.95}}
        """
        response = gpt_with_retry(client,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        if response:
            result = json.loads(response.choices[0].message.content.strip())
            gl_guess = result.get("gl_account")
            confidence = result.get("confidence", 0)
            if confidence >= 0.9:
                cursor.execute("INSERT INTO vendor_gl (vendor, corrected_vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, ?, 1)",
                              (vendor_cleaned, vendor, gl_guess, datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
                return gl_guess
    except Exception as e:
        print("[ERROR] GPT match failed:", e)

    for _, row in coa_df.iterrows():
        sample_vendors = str(row['Sample Vendors']).lower().split(',')
        if any(sample.strip() in vendor_cleaned for sample in sample_vendors):
            return row['GL Account']

    return None

def ai_ocr_from_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    try:
        response = gpt_with_retry(client,
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You're a document assistant. Extract visible transactions from this image."},
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
        print("[ERROR] Vision OCR failed:", e)
        return ""

def extract_text_from_image(image):
    try:
        import pytesseract
        return pytesseract.image_to_string(image)
    except Exception as e:
        print("[WARNING] pytesseract failed, using GPT vision:", e)
        return ai_ocr_from_image(image)

st.set_page_config(page_title="Bank Statement Categorizer", layout="centered")

with st.container():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("data/ValiantIconWhite.png", width=250)
    st.title("Bank Statement Categorizer")
    st.markdown("Upload PDF statements to categorize transactions.")
    st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("\U0001F4C4 Upload Bank Statement (PDF)", type=["pdf"])

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
                date_match = re.match(r'(\d{2}/\d{2})\s+', line)
                if date_match:
                    full_line = line
                    if j + 1 < len(lines):
                        full_line += " " + lines[j + 1]
                    match = re.match(r'(\d{2}/\d{2})[^\d]*(.*?)\s+([-+]?\$?\d+[,.]\d{2})', full_line)
                    if match:
                        date_raw, description, amount_str = match.groups()
                        try:
                            parsed_date = datetime.strptime(date_raw + f"/{datetime.now().year}", "%m/%d/%Y")
                        except:
                            continue
                        amount_str = amount_str.replace("$", "").replace(",", "")
                        try:
                            amount = float(amount_str)
                        except:
                            amount = None
                        vendor = clean_vendor(description)
                        category = match_category(vendor, coa_df, cursor)
                        status = "Auto-Categorized" if category else "Uncategorized"
                        reason = "Matched" if category else "Needs Review"

                        transactions.append({
                            "Date": parsed_date.strftime("%Y-%m-%d"),
                            "Description": description.strip(),
                            "Amount": amount,
                            "Vendor": vendor,
                            "Category": category,
                            "Status": status,
                            "Reason": reason
                        })

        progress_bar.empty()
        df = pd.DataFrame(transactions)
        st.subheader("\U0001F50D Extracted Transactions")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

        for _, row in edited_df.iterrows():
            if row["Vendor"] and row["Category"]:
                vendor = row["Vendor"].strip()
                vendor_cleaned = vendor.lower()
                gl = row["Category"]
                cursor.execute("""
                    INSERT OR REPLACE INTO vendor_gl (
                        vendor, corrected_vendor, gl_account, last_used, usage_count
                    ) VALUES (?, ?, ?, ?, 
                        COALESCE((SELECT usage_count FROM vendor_gl WHERE corrected_vendor = ?), 0) + 1)
                """, (vendor_cleaned, vendor, gl, datetime.now().strftime('%Y-%m-%d'), vendor_cleaned))
                conn.commit()

        st.download_button("Download CSV", data=edited_df.to_csv(index=False), file_name="categorized_transactions.csv", mime="text/csv")