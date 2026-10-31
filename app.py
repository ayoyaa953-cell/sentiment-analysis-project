
import os
import time
import gradio as gr
from google import genai
from google.genai import types
from google.genai.errors import APIError

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text."

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=f"""
                Analyze the sentiment of this text:
                {text.strip()}

                Return exactly one label:
                Positive, Negative, or Neutral.
                """,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="low"
                    ),
                    max_output_tokens=10
                )
            )

            result = (response.text or "").strip().lower()

            if result in ["positive", "negative", "neutral"]:
                return result.capitalize()

            return "Unable to classify sentiment."

        except APIError as e:
            if getattr(e, "code", None) in [429, 500, 502, 503, 504] and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            return "Gemini API error. Please try again."

        except Exception:
            return "Unexpected error. Please try again."

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

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)


