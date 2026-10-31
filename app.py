
import os
import time
import gradio as gr
from google import genai
from google.genai.errors import ServerError

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=f"""
                Analyze the sentiment of this text:
                {text}

                Return only one label:
                Positive, Negative, or Neutral.
                """
            )
            return response.text.strip()

        except ServerError as e:
            if e.status_code == 503 and attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return "Service temporarily unavailable. Please try again."

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter your text here...",
        label="Your Text"
    ),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="AI Sentiment Analysis System",
    description="Enter a sentence and AI will analyze its sentiment."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
