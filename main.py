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
coa_df.dropna(how='all', inplace=True)  # Drop empty rows
coa_df.columns = [col.strip() for col in coa_df.columns]
coa_df = coa_df[['GL Account', 'Account Name', 'Sample Vendors']].dropna(subset=['GL Account', 'Sample Vendors'])

# Matching function to categorize transactions
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content.strip())
        gl_guess = result.get("gl_account")
        confidence = result.get("confidence", 0)

        if confidence >= 0.9:
            c.execute("INSERT INTO vendor_gl (vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, 1)",
                      (description_lower, gl_guess, datetime.now().strftime('%Y-%m-%d')))
            conn.commit()
            return gl_guess
    except Exception as e:
        print("OpenAI error:", e)

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

# Streamlit App
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

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                progress_bar.progress((i + 1) / len(pdf.pages))
                text = page.extract_text()
                if not text:
                    img = page.to_image(resolution=300)
                    image = img.original
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

        st.markdown("**🎓 Manual Review & Override**")
        manual_inputs = {}
        for idx, row in df.iterrows():
            current_cat = row['Category']
            vendor = row['Vendor']
            options = [""] + list(coa_df['GL Account'].unique())
            default_index = options.index(current_cat) if current_cat in options else 0
            selected_gl = st.selectbox(f"{vendor} ({row['Description']})", options=options, index=default_index, key=f"manual_{idx}")
            manual_inputs[idx] = selected_gl

        if st.button("💾 Save Changes"):
            for idx, new_gl in manual_inputs.items():
                if new_gl:
                    df.at[idx, 'Category'] = new_gl
                    df.at[idx, 'Status'] = "Manually Updated"
                    df.at[idx, 'Reason'] = "Manual Override"
                    vendor_lower = df.at[idx, 'Vendor'].lower()
                    c.execute("INSERT OR REPLACE INTO vendor_gl (vendor, gl_account, last_used, usage_count) VALUES (?, ?, ?, COALESCE((SELECT usage_count FROM vendor_gl WHERE vendor = ?), 0) + 1)",
                              (vendor_lower, new_gl, datetime.now().strftime('%Y-%m-%d'), vendor_lower))
            conn.commit()
            st.success("Manual changes saved.")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name='categorized_transactions.csv',
            mime='text/csv'
        )
else:
    st.info("Please upload a PDF bank statement to begin.")
