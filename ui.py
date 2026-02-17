import gradio as gr
import requests
import json

# Configuration
API_URL = "http://localhost:5000/query"

def ask_api(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    print(f"Sending query to {API_URL}: {question}")
    
    try:
        # Send POST request to Flask API
        response = requests.post(API_URL, json={"question": question})
        
        if response.status_code == 200:
            data = response.json()
            answer_text = data.get("answer", "No answer provided.")
            sources = data.get("sources", [])
            
            # Format sources for display
            sources_text = "**Sources:**\n"
            if not sources:
                sources_text += "No sources cited."
            else:
                for s in sources:
                    filename = s.get('filename', 'Unknown')
                    page = s.get('page_num', '?')
                    sources_text += f"- {filename} (Page {page})\n"
            
            return answer_text, sources_text
        else:
            return f"Error: Server returned status code {response.status_code}\n{response.text}", ""
            
    except requests.exceptions.ConnectionError:
        return "Connection Failed. Is the server running? (Run `python server.py`)", ""
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

# Define Gradio Interface
with gr.Blocks(title="RAG Pro Max (Client)") as demo:
    gr.Markdown("# 🤖 RAG Pro Max: Document Assistant (Client Mode)")
    gr.Markdown("Ensure `server.py` is running in a separate terminal!")
    
    with gr.Row():
        with gr.Column():
            q_input = gr.Textbox(label="Your Question", placeholder="Ask about your documents...")
            submit_btn = gr.Button("Ask", variant="primary")
            
        with gr.Column():
            a_output = gr.Markdown(label="Answer")
            s_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=ask_api,
        inputs=[q_input],
        outputs=[a_output, s_output]
    )
    
    # Allow pressing Enter to submit
    q_input.submit(
        fn=ask_api,
        inputs=[q_input],
        outputs=[a_output, s_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
