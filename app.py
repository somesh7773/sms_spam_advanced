import gradio as gr
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Lists to store messages
inbox_messages = []
spam_messages = []

# Predict and store
def classify_sms(message):
    if not message.strip():
        return gr.update(), gr.update()

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]

    if prediction == "spam":
        spam_messages.append(message)
    else:
        inbox_messages.append(message)

    return "\n\n".join(inbox_messages[-10:]), "\n\n".join(spam_messages[-10:])

# UI layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📩 Real-Time SMS Spam Filter")
    gr.Markdown("Enter a message below. Spam will be separated automatically.")

    with gr.Row():
        with gr.Column(scale=2):
            message_input = gr.Textbox(label="Enter SMS Text", placeholder="Type a message...")
            classify_btn = gr.Button("🔍 Check & Store")

        with gr.Column():
            inbox_box = gr.Textbox(label="✅ Inbox", lines=10, interactive=False)
            spam_box = gr.Textbox(label="🚫 Spam Folder", lines=10, interactive=False)

    classify_btn.click(fn=classify_sms, inputs=message_input, outputs=[inbox_box, spam_box])

# Launch app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
