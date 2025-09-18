# app.py
import os
import json
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_AVAILABLE = False


def make_groq_client():
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None


def local_fallback(transcript: str):
    """Simple offline summarizer + sentiment (naive)."""
    s = transcript.strip()
    sentences = re.split(r'[.?!]\s+', s)
    summary = sentences[0].strip() if sentences else " ".join(s.split()[:12]) + "..."
    
    # naive sentiment
    t = s.lower()
    neg = sum(t.count(w) for w in ["not", "failed", "frustrat", "angry", "charged", "refund", "problem", "issue", "complain", "delay"])
    pos = sum(t.count(w) for w in ["thank", "thanks", "great", "happy", "good", "satisfied"])
    sentiment = "Negative" if neg > pos else "Positive" if pos > neg else "Neutral"
    return summary, sentiment


def call_groq_for_json(client, transcript: str, model_name: str = "llama-3.1-8b-instant"):
    system = {
        "role": "system",
        "content": (
            "You are an assistant that MUST return strictly valid JSON only. "
            "Given a customer call transcript, return a JSON object with keys: "
            "'summary' (a concise 2-3 sentence summary) and 'sentiment' (one of: Positive, Neutral, Negative). "
            "Return *only* the JSON object and nothing else."
        )
    }
    user = {
        "role": "user",
        "content": f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\nRespond with the JSON object described."
    }

    resp = client.chat.completions.create(messages=[system, user], model=model_name, temperature=0)
    generated = resp.choices[0].message.content if hasattr(resp.choices[0].message, "content") else str(resp)

    try:
        parsed = json.loads(generated)
        return parsed.get("summary", "").strip(), parsed.get("sentiment", "").strip(), generated
    except Exception:
        m = re.search(r"\{.*\}", generated, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed.get("summary", "").strip(), parsed.get("sentiment", "").strip(), generated
            except Exception:
                pass
    return generated.strip(), "Unknown", generated


# --- Streamlit UI ---
st.set_page_config(page_title="Customer Call Analyzer", layout="wide")
st.title("📞 Customer Call Analyzer")

st.markdown(
    "Paste a call transcript, click **Analyze**, and the app will show a 2–3 sentence summary, "
    "with its sentiment (Positive, Neutral, Negative)."
)

# Default Groq model internally
default_model_name = "llama-3.1-8b-instant"

# Sample transcripts with mixed sentiment
sample = st.selectbox("Try a sample transcript", options=[
    "— Select sample —",
    "Hi, I tried to book a slot yesterday but the payment failed and I was charged twice. I’m really frustrated and want a refund immediately.",
    "Good afternoon, I recently ordered a laptop from your website. It arrived yesterday, but the screen is cracked, and I can’t use it. Please help me replace it quickly.",
    "Hello, I’m trying to log into my account but I keep getting an error message that my password is incorrect, even though I just reset it yesterday.",
    "I booked a flight through your app last week. Today I got an email saying my booking was cancelled without my consent. I urgently need this resolved.",
    "Hi, I’m a long-time customer, but lately your delivery service has been very slow. My last two orders arrived late by more than a week. I need reassurance this won’t happen again.",
    "Hello, I purchased headphones last month, and they stopped working after only a few days. I tried troubleshooting, but nothing helps. Can I get a replacement or refund?",
    "Good morning, I received my order yesterday and everything is perfect. I really appreciate the fast delivery and excellent packaging. Thank you!",
    "Hi, I just wanted to say that the support team helped me resolve my issue very quickly. I am happy with the service and will continue using your platform.",
    "Hello, I called earlier about my car rental booking, but the representative didn’t provide a clear resolution. I would like to know the next steps to complete the process.",
    "Good evening, I was double charged for my subscription this month. I only need one active subscription, so please cancel the duplicate charge and issue a refund."
])

transcript = st.text_area("Transcript", value=(sample if sample != "— Select sample —" else ""), height=200)

if st.button("Analyze"):
    if not transcript.strip():
        st.warning("⚠️ Please paste or type a transcript before analyzing.")
    else:
        # Decide whether to use Groq or local fallback
        use_offline = (not GROQ_AVAILABLE) or (not GROQ_API_KEY)
        if use_offline:
            summary, sentiment = local_fallback(transcript)
            used = "LOCAL_FALLBACK"
        else:
            client = make_groq_client()
            if not client:
                st.warning("Groq client not available. Using local fallback.")
                summary, sentiment = local_fallback(transcript)
                used = "LOCAL_FALLBACK"
            else:
                try:
                    summary, sentiment, _ = call_groq_for_json(client, transcript, model_name=default_model_name)
                    used = "GROQ"
                except Exception as e:
                    st.warning(f"Error calling Groq: {e}. Using local fallback.")
                    summary, sentiment = local_fallback(transcript)
                    used = "LOCAL_FALLBACK"

        # Save to CSV
        csv_path = "call_analysis.csv"
        df_row = pd.DataFrame([{"Transcript": transcript, "Summary": summary, "Sentiment": sentiment}])
        df_row.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False, encoding="utf-8")

        # Display in UI
        st.subheader("Transcript")
        st.code(transcript)
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Sentiment")
        st.write(sentiment)
        st.success(f"✅ Saved to `{csv_path}`. Used: {used}")
