import streamlit as st
import pandas as pd
import sqlite3
import tempfile
import os
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from datetime import datetime
import re

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
coa_df.dropna(how='all', inplace=True)  # Drop empty rows
coa_df.columns = [col.strip() for col in coa_df.columns]
coa_df = coa_df[['GL Account', 'Account Name', 'Sample Vendors']].dropna(subset=['GL Account', 'Sample Vendors'])

# Matching function to categorize transactions
def match_category(description, coa_df):
    if not isinstance(description, str):
        return None

    description_lower = description.lower()

    # First, check the vendor memory DB
    c.execute("SELECT gl_account FROM vendor_gl WHERE vendor = ?", (description_lower,))
    result = c.fetchone()
    if result:
        c.execute("UPDATE vendor_gl SET usage_count = usage_count + 1, last_used = ? WHERE vendor = ?", (datetime.now().strftime('%Y-%m-%d'), description_lower))
        conn.commit()
        return result[0]

    # Then, check COA for sample vendor match
    for _, row in coa_df.iterrows():
        sample_vendors = str(row['Sample Vendors']).lower().split(',')
        for vendor in sample_vendors:
            if vendor.strip() in description_lower:
                # Save to memory for future auto-categorization
                c.execute("INSERT INTO vendor_gl (vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, 1)",
                          (description_lower, row['GL Account'], datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
                return row['GL Account']

    return None

# Streamlit App
st.set_page_config(page_title="Bank Statement Categorizer", layout="centered")

# Centered header with logo
with st.container():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("data/ValiantIconWhite.png", width=250)
    st.title("Bank Statement Categorizer")
    st.markdown("Upload PDF bank statements and categorize transactions using your firm’s Chart of Accounts.")
    st.markdown("</div>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📄 Upload Bank Statement (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        images = convert_from_bytes(uploaded_file.read(), fmt='jpeg')
        transactions = []

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if "intentionally left blank" in text.lower() or len(text.strip()) < 20:
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]

            for j, line in enumerate(lines):
                date_match = re.match(r'(\d{2}/\d{2})\s+(Card Purchase|Online Transfer|Recurring Card Purchase|ATM Withdrawal)?', line)
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
                        reason = "Matched from Memory/COA" if category else "Manual Categorization Skip"

                        transactions.append({
                            "Date": parsed_date.strftime("%Y-%m-%d"),
                            "Description": description.strip(),
                            "Amount": amount,
                            "Vendor": description.strip(),
                            "Category": category,
                            "Status": status,
                            "Reason": reason
                        })

        df = pd.DataFrame(transactions)
        st.subheader("🔍 Extracted Transactions")
        st.dataframe(df if not df.empty else "empty")

        # Manual Categorization UI
        uncategorized_vendors = df[df['Category'].isna()]['Vendor'].unique()
        if len(uncategorized_vendors) > 0:
            st.markdown("**🎓 Manual Categorization**")
            manual_inputs = {}
            for vendor in uncategorized_vendors:
                gl_account = st.selectbox(f"Assign GL Account to: {vendor}", options=coa_df['GL Account'].unique(), key=vendor)
                manual_inputs[vendor] = gl_account

            if st.button("Save Manual Mappings"):
                for vendor, gl in manual_inputs.items():
                    c.execute("INSERT OR REPLACE INTO vendor_gl (vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, COALESCE((SELECT usage_count FROM vendor_gl WHERE vendor = ?), 0) + 1)",
                              (vendor.lower(), gl, datetime.now().strftime('%Y-%m-%d'), vendor.lower()))
                conn.commit()
                st.success("Manual mappings saved. Please re-upload the PDF to refresh categorization.")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name='categorized_transactions.csv',
            mime='text/csv'
        )
else:
    st.info("Please upload a PDF bank statement to begin.")
