
import os
import time
import re
from datetime import datetime

import gradio as gr
from google import genai
from google.genai import types
from google.genai.errors import APIError


# Gemini configuration
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL_NAME = "gemini-3.8-flash"

LABELS = ["Positive", "Negative", "Neutral"]


# Sentiment analysis function
def analyze_sentiment(text):

    if not text or not text.strip():
        return "Please enter some text.", 0

    start_time = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"""
            Classify the sentiment of this text.

            Text: {text.strip()}

            Return exactly one word:
            Positive, Negative, or Neutral.
            """,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                ),
                max_output_tokens=10
            )
        )

        result = response.text or ""

        match = re.search(
            r"\b(positive|negative|neutral)\b",
            result,
            re.IGNORECASE
        )

        elapsed = round(
            time.perf_counter() - start_time, 2
        )

        if match:
            return match.group(1).capitalize(), elapsed

        return "Unable to classify sentiment.", elapsed

    except APIError as error:
        print(f"Gemini API Error: {error}")
        return "Gemini service is temporarily unavailable.", 0

    except Exception as error:
        print(f"Unexpected Error: {error}")
        return "An unexpected error occurred.", 0


# Update analysis history and dashboard
def process_analysis(text, history):

    history = history or []

    result, elapsed = analyze_sentiment(text)

    if result not in LABELS:
        return (
            result,
            history,
            history,
            create_dashboard(history)
        )

    record = {
        "text": text.strip(),
        "sentiment": result,
        "time": elapsed,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    new_history = history + [record]

    return (
        f"## {result}\n\nAnalysis time: {elapsed} seconds",
        new_history,
        new_history,
        create_dashboard(new_history)
    )


# Dashboard statistics
def create_dashboard(history):

    history = history or []

    total = len(history)

    positive = sum(
        item["sentiment"] == "Positive"
        for item in history
    )

    negative = sum(
        item["sentiment"] == "Negative"
        for item in history
    )

    neutral = sum(
        item["sentiment"] == "Neutral"
        for item in history
    )

    if total == 0:
        return """
        ## Analytics Dashboard

        No analyses yet. Start by analyzing some text.
        """

    return f"""
    ## Analytics Dashboard

    ### Total Analyses: {total}

    | Sentiment | Count |
    |---|---:|
    | Positive | {positive} |
    | Negative | {negative} |
    | Neutral | {neutral} |

    ### Sentiment Distribution

    - Positive: {positive / total * 100:.1f}%
    - Negative: {negative / total * 100:.1f}%
    - Neutral: {neutral / total * 100:.1f}%
    """


# Clear history
def clear_history():
    empty_history = []

    return (
        empty_history,
        [],
        create_dashboard(empty_history)
    )


# Website design
css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 25px;
}

"""


# Main website
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
    title="AI Sentiment Intelligence Platform"
) as demo:

    history_state = gr.State([])

    gr.Markdown(
        """
        # AI Sentiment Intelligence Platform

        ### Understand emotions in text using Artificial Intelligence.
        """,
        elem_id="main-title"
    )

    with gr.Tabs():

        # Home Page
        with gr.Tab("Home"):

            gr.Markdown(
                """
                ## Welcome to AI Sentiment Intelligence

                This platform uses Google's Gemini AI
                to understand the emotional meaning of text.

                ### Main Features

                - AI-powered sentiment analysis
                - Positive, Negative and Neutral classification
                - Interactive analytics dashboard
                - Analysis history

                **Start analyzing your text today!**
                """
            )

        # Analyze Page
        with gr.Tab("Analyze"):

            gr.Markdown("## Sentiment Analysis")

            text_input = gr.Textbox(
                lines=5,
                label="Enter Your Text",
                placeholder="Write your sentence here..."
            )

            analyze_button = gr.Button(
                "Analyze Sentiment",
                variant="primary"
            )

            result_output = gr.Markdown(
                "Your result will appear here."
            )

            analyze_button.click(
                fn=process_analysis,
                inputs=[text_input, history_state],
                outputs=[
                    result_output,
                    history_state,
                    history_state,
                    dashboard_output
                ]
            )

        # Dashboard Page
        with gr.Tab("Dashboard"):

            gr.Markdown("## Analytics Dashboard")

            dashboard_output = gr.Markdown(
                create_dashboard([])
            )

        # History Page
        with gr.Tab("History"):

            gr.Markdown("## Analysis History")

            history_table = gr.JSON(
                label="Previous Analyses"
            )

            clear_button = gr.Button(
                "Clear History",
                variant="stop"
            )

            clear_button.click(
                fn=clear_history,
                inputs=[],
                outputs=[
                    history_state,
                    history_table,
                    dashboard_output
                ]
            )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)



