import streamlit as st
import pandas as pd
import sqlite3
import tempfile
import os
import pdfplumber
from PIL import Image
import pytesseract
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
client = OpenAI()

# Initialize SQLite database for vendor GL mapping
if not os.path.exists("database"):
    os.makedirs("database")
conn = sqlite3.connect("database/vendor_gl.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS vendor_gl (
                vendor TEXT,
                gl_account TEXT,
                last_used TEXT,
                usage_count INTEGER
            )''')
conn.commit()

# Load pre-defined Chart of Accounts from Excel file
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
                wait_time = backoff + random.uniform(0, 1)
                print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                backoff *= 2
            else:
                print("OpenAI error:", e)
                break
    return None

def match_category(description, coa_df):
    if not isinstance(description, str):
        return None
    description_lower = description.lower()

    try:
        prompt = f"""
        You are an accountant. Based on the following chart of accounts, choose the best GL Account for the transaction.

        Transaction Description: {description}

        Chart of Accounts:
        {coa_df[['GL Account', 'Account Name']].to_string(index=False)}

        Respond only with a JSON object like this:
        {{ "gl_account": "Your Suggested GL Account", "confidence": 0.95 }}
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
                c.execute("INSERT INTO vendor_gl (vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, 1)",
                          (description_lower, gl_guess, datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
                return gl_guess
    except Exception as e:
        print("OpenAI fallback error:", e)

    c.execute("SELECT gl_account FROM vendor_gl WHERE vendor = ?", (description_lower,))
    result = c.fetchone()
    if result:
        c.execute("UPDATE vendor_gl SET usage_count = usage_count + 1, last_used = ? WHERE vendor = ?", (datetime.now().strftime('%Y-%m-%d'), description_lower))
        conn.commit()
        return result[0]

    for _, row in coa_df.iterrows():
        sample_vendors = str(row['Sample Vendors']).lower().split(',')
        for vendor in sample_vendors:
            if vendor.strip() in description_lower:
                c.execute("INSERT INTO vendor_gl (vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, 1)",
                          (description_lower, row['GL Account'], datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
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
                {"role": "system", "content": "You're a document extraction assistant. Extract readable transaction text."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract visible transactions from this bank statement image. Return as plain text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        if response:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print("AI OCR fallback error:", e)
    return ""

st.set_page_config(page_title="Bank Statement Categorizer", layout="centered")

with st.container():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("data/ValiantIconWhite.png", width=250)
    st.title("Bank Statement Categorizer")
    st.markdown("Upload PDF bank statements and categorize transactions using your firm’s Chart of Accounts.")
    st.markdown("</div>", unsafe_allow_html=True)

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

            text = pytesseract.image_to_string(image)

            if not text or len(text.strip()) < 20:
                text = ai_ocr_from_image(image)

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
                            parsed_date = datetime.strptime(date_raw + "/2018", "%m/%d/%Y")
                        except:
                            continue
                        amount_str = amount_str.replace("$", "").replace(",", "")
                        try:
                            amount = float(amount_str)
                        except:
                            amount = None
                        category = match_category(description, coa_df)
                        status = "Auto-Categorized" if category else "Uncategorized"
                        reason = "Matched from GPT / Memory / COA" if category else "Manual Categorization Skip"

                        transactions.append({
                            "Date": parsed_date.strftime("%Y-%m-%d"),
                            "Description": description.strip(),
                            "Amount": amount,
                            "Vendor": description.strip(),
                            "Category": category,
                            "Status": status,
                            "Reason": reason
                        })

        progress_bar.empty()
        df = pd.DataFrame(transactions)
        st.subheader("🔍 Extracted Transactions")
        st.dataframe(df if not df.empty else "empty")
