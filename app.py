

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader



#  Train model 

MODEL_FILE = "disease_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
DATA_FILE = "Training.csv"

@st.cache_resource(show_spinner=False)
def train_or_load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        model = joblib.load(MODEL_FILE)
        le = joblib.load(ENCODER_FILE)
        df = pd.read_csv(DATA_FILE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return model, le, df

    st.warning("Model not found — training new model from Training.csv...")

    # Load Dataset
    df = pd.read_csv(DATA_FILE)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.fillna(0)

    X = df.drop(columns=['prognosis'])
    y = df['prognosis']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.info(f" Model trained successfully! Accuracy: {acc * 100:.2f}%")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)

    return model, le, df


# Load model, encoder, and symptom columns

st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title(" Disease Prediction from Symptoms")

model, label_encoder, df = train_or_load_model()
symptom_cols = df.columns[:-1].tolist()


#  Session history setup

if "history" not in st.session_state:
    st.session_state.history = []

#  User Interface
left, right = st.columns([1, 1])

with left:
    st.subheader("Select Symptoms")
    selected = st.multiselect(
        "Choose symptoms (you can select many):",
        options=symptom_cols,
        max_selections=15
    )

    st.write("Or type symptoms (comma-separated):")
    text_input = st.text_input("e.g., fever, cough, headache")

    st.markdown("**Optional:** Capture a photo (visible symptom).")
    photo_bytes = st.camera_input("Capture Photo")

    if st.button(" Predict Disease"):
        if text_input:
            tokens = [t.strip().lower() for t in text_input.split(",") if t.strip()]
            matched = []
            for tok in tokens:
                for col in symptom_cols:
                    if tok == col.lower() or tok in col.lower() or col.lower() in tok:
                        matched.append(col)
            for m in matched:
                if m not in selected:
                    selected.append(m)

        if not selected:
            st.warning("Please select or enter symptoms.")
        else:
            input_vec = np.zeros(len(symptom_cols), dtype=int)
            for s in selected:
                if s in symptom_cols:
                    input_vec[symptom_cols.index(s)] = 1

            input_df = pd.DataFrame([input_vec], columns=symptom_cols)
            pred_encoded = model.predict(input_df)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            confidence = None
            top_preds = [(pred_label, None)]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = float(np.max(probs)) * 100.0
                top_idx = np.argsort(probs)[::-1][:3]
                top_preds = [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_idx]

            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disease": pred_label,
                "confidence": round(confidence, 2) if confidence else None,
                "symptoms": selected.copy(),
                "photo": photo_bytes.getvalue() if photo_bytes else None
            }
            st.session_state.history.append(entry)

            with right:
                st.subheader("Prediction Result")
                st.success(f"Predicted Disease: **{pred_label}**")
                if confidence:
                    st.info(f"Confidence: {confidence:.2f}%")
                st.write("Top Predictions:")
                for lbl, p in top_preds:
                    st.write(f"- {lbl} — {p:.2%}" if p else f"- {lbl}")

with right:
    st.subheader("Selected Symptoms")
    st.write(", ".join(selected) if selected else "None selected yet.")

    st.markdown("---")
    st.write("""
    This model uses a RandomForest trained on the Training.csv dataset.
    Predictions are based on binary symptom vectors.
    """)

# History & Charts
st.markdown("---")
st.header(" Prediction History & Reports")

history_df = pd.DataFrame(st.session_state.history)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("History Table")
    if not history_df.empty:
        df_show = history_df.drop(columns=["photo"], errors="ignore")
        st.dataframe(df_show.sort_values("timestamp", ascending=False))
        csv_data = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(" Download CSV", csv_data, "prediction_history.csv", "text/csv")
    else:
        st.info("No predictions yet.")

with col2:
    st.subheader("Charts")
    if not history_df.empty and "confidence" in history_df.columns:
        conf_df = history_df.dropna(subset=["confidence"]).copy()
        if not conf_df.empty:
            conf_df["timestamp_dt"] = pd.to_datetime(conf_df["timestamp"])
            conf_df = conf_df.sort_values("timestamp_dt")
            st.line_chart(conf_df.set_index("timestamp_dt")["confidence"], height=250)
        dist = history_df["disease"].value_counts()
        st.bar_chart(dist, height=250)
    else:
        st.info("Charts will appear after predictions.")


# PDF Report
st.markdown("---")
st.subheader("Generate PDF Report")

def generate_pdf_bytes(entries, include_photos=False):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "Disease Prediction Report")
    y -= 30
    c.setFont("Helvetica", 10)

    for e in entries:
        text = f"{e['timestamp']} — {e['disease']} ({e['confidence']}%)"
        c.drawString(40, y, text)
        y -= 14
        c.drawString(60, y, f"Symptoms: {', '.join(e['symptoms'])}")
        y -= 14
        if include_photos and e.get("photo"):
            try:
                img = ImageReader(io.BytesIO(e["photo"]))
                img_w, img_h = 120, 90
                c.drawImage(img, width - img_w - 40, y - img_h + 14, width=img_w, height=img_h)
                y -= (img_h + 6)
            except:
                pass
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

include_photos = st.checkbox("Include photos in PDF", value=False)
if st.button(" Generate PDF"):
    if history_df.empty:
        st.warning("No predictions to include in report.")
    else:
        pdf_bytes = generate_pdf_bytes(st.session_state.history[::-1], include_photos)
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

st.caption(" Photos and data are stored only in this session — not saved permanently.")

