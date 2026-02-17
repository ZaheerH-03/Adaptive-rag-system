"""
RAG Pro Max — Chatbot UI
Thin client that talks to server.py via HTTP.

Run:
  Terminal 1:  python server.py
  Terminal 2:  python ui.py
"""

import gradio as gr
import requests

API_URL = "http://localhost:5000"


# ── Chat callback ──────────────────────────────────────────────────
def respond(message, history):
    """
    message : str               — user's latest message
    history : list[list[str]]   — [[user_msg, bot_msg], ...]
    """
    if not message.strip():
        return "Please type a question."

    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": message},
            timeout=300,
        )

        if resp.status_code != 200:
            return f"⚠️ Server error ({resp.status_code})"

        data = resp.json()
        answer = data.get("answer", "No answer.")
        sources = data.get("sources", [])

        # Append source citations
        if sources:
            answer += "\n\n---\n📚 **Sources:**\n"
            for i, s in enumerate(sources, 1):
                fname = s.get("filename", "?")
                page = s.get("page_num")
                slide = s.get("slide_num")
                loc = f"Page {page}" if page else (f"Slide {slide}" if slide else "")
                answer += f"{i}. *{fname}*"
                if loc:
                    answer += f" ({loc})"
                answer += "\n"

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "❌ **Cannot reach the server.**\n\n"
            "Run `python server.py` in another terminal first."
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ── Custom CSS ─────────────────────────────────────────────────────
CSS = """
/* page background */
body {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
}
.gradio-container {
    background: transparent !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* title gradient */
h1 {
    text-align: center !important;
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2.2rem !important;
}

/* chatbot area */
.chatbot {
    background: rgba(15, 15, 30, 0.5) !important;
    border: 1px solid rgba(102, 126, 234, 0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
}

/* user bubble */
.message-row.user .message-bubble-border {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 14px !important;
}

/* bot bubble */
.message-row.bot .message-bubble-border {
    background: rgba(30, 30, 55, 0.85) !important;
    border: 1px solid rgba(102, 126, 234, 0.12) !important;
    border-radius: 14px !important;
}

/* input box */
textarea {
    background: rgba(20, 20, 40, 0.85) !important;
    border: 1px solid rgba(102, 126, 234, 0.25) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
}
textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}

/* submit button */
button.primary {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
button.primary:hover {
    opacity: 0.9 !important;
}
"""

# ── Theme ──────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
)


# ── Build the app ──────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=CSS, title="RAG Pro Max") as demo:

    gr.ChatInterface(
        fn=respond,
        title="RAG Pro Max 🤖",
        description="Ask me anything about your indexed documents.",
        examples=[
            "What are the different types of Organizational Behaviour models?",
            "Explain motivation theories.",
            "Tell me about magpie sensing and their solution.",
        ],
    )


if __name__ == "__main__":
    demo.launch(share=False)
