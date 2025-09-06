import streamlit as st
import pandas as pd
import joblib
import re

model = joblib.load("spam_model.pkl")

suspicious_words = [
    "win", "prize", "urgent", "click", "account", "bank", "password",
    "verify", "update", "free", "offer", "reward", "limited", "security",
    "login", "suspend", "confirm", "risk", "payment", "invoice", "credit"
]

st.set_page_config(
    page_title="Spam & Phishing Email Detector",
    page_icon="📧",
    layout="wide"
)

st.sidebar.title("📧 Spam & Phishing Detector")
st.sidebar.info(
    """
    Detect spam and phishing emails using AI.
    
    **Features:**
    - Single email prediction
    - Batch prediction via CSV
    - Suspicious word highlighting
    - Risk meter
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Example Emails")
st.sidebar.write('"You won $1,000,000! Click here to claim!" → spam')
st.sidebar.write('"Meeting at 3 PM tomorrow." → safe')

st.title("📧 AI-Powered Spam & Phishing Detector")
st.info(
    """
    **How to use:**  
    1. Paste your email in the box and click Predict.  
    2. Or upload a CSV with a column `text` for batch predictions.  
    3. Suspicious words are highlighted in yellow.  
    4. Risk meter shows overall email risk.
    """
)
st.markdown("---")

st.header("✉️ Single Email Prediction")
col1, col2 = st.columns([2, 1])

with col1:
    email_text = st.text_area("Paste your email here", height=200)

with col2:
    if st.button("Predict Single Email"):
        if email_text.strip() == "":
            st.warning("⚠️ Please enter some email text!")
        else:
            prediction = model.predict([email_text])[0]
            confidence = model.predict_proba([email_text])[0].max() * 100 if hasattr(model, "predict_proba") else None
            
            if prediction.lower() == "spam":
                st.markdown(f"""
                <div style='background-color:#FF6961; padding:15px; border-radius:10px; color:white; font-weight:bold; text-align:center'>
                    ⚠️ Prediction: SPAM {f'({confidence:.2f}%)' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#4CAF50; padding:15px; border-radius:10px; color:white; font-weight:bold; text-align:center'>
                    ✅ Prediction: SAFE {f'({confidence:.2f}%)' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
            
            def highlight_words(text, words):
                for word in words:
                    text = re.sub(f"(?i)\\b({word})\\b", r'<mark style="background-color: #FFD700">\1</mark>', text)
                return text

            st.markdown("**Highlighted suspicious words:**")
            st.markdown(highlight_words(email_text, suspicious_words), unsafe_allow_html=True)

            count = sum(len(re.findall(f"(?i)\\b{word}\\b", email_text)) for word in suspicious_words)
            st.info(f"⚠️ Suspicious words detected: {count}")

            max_count = 10
            risk_percent = min(100, (count / max_count) * 100)
            color = "#4CAF50" if risk_percent < 30 else "#FFA500" if risk_percent < 70 else "#FF4500"

            st.markdown(f"""
            <div style='background-color:#e0e0e0; border-radius:10px; padding:3px; width:100%'>
              <div style='width:{risk_percent}%; background-color:{color}; padding:10px 0; border-radius:10px; text-align:center; color:white; font-weight:bold'>
                Risk: {risk_percent:.0f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

st.header("📂 Batch Email Prediction (CSV)")
st.info("Upload a CSV file with a column `text`. Results will include predictions and highlighted suspicious words.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("❌ CSV must have a column named 'text'")
    else:
        df['prediction'] = model.predict(df['text'])
        
        def highlight_val(val):
            return 'color: red; font-weight: bold' if any(word.lower() in val.lower() for word in suspicious_words) else ''

        st.success("✅ Predictions complete!")
        st.dataframe(df.style.applymap(highlight_val, subset=['text']))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

st.markdown("---")
st.caption("Developed by Ojaswita Dhar | AI Spam & Phishing Email Detector")

