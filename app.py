import gradio as gr
from fastapi import FastAPI, Request
import joblib
import os

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to classify SMS
def classify_sms(sms):
    vect = vectorizer.transform([sms])
    pred = model.predict(vect)
    return "✅ Not Spam" if pred[0] == "ham" else "🚫 Spam"

# Gradio Interface
interface = gr.Interface(
    fn=classify_sms,
    inputs="text",
    outputs="text",
    title="SMS Spam Filter",
    description="Enter an SMS message to check if it's spam or not."
)

# FastAPI app for IFTTT or webhook
app = FastAPI()

@app.post("/receive-sms")
async def receive_sms(request: Request):
    data = await request.json()
    sms_text = data.get("sms", "")
    result = classify_sms(sms_text)
    print(f"Received via IFTTT: {sms_text} → {result}")
    return {"status": "received", "result": result}

# Run Gradio on Render (do NOT use app.launch)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    interface.launch(server_name="0.0.0.0", server_port=port)
