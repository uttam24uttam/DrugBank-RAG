# ui.py
import gradio as gr
import requests

API_URL = "https://suggestibly-unstrewn-lashaun.ngrok-free.dev/chat"


def ask_backend(question):
    if not question:
        return "Please enter a question.", "", "<i>No chunks</i>"

    try:
        resp = requests.post(API_URL, json={"question": question}, timeout=200)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error contacting backend: {e}", "", ""

    answer = data.get("answer", "")
    drug = data.get("drug_extracted", "")
    chunks = data.get("chunks_used", [])

    if not chunks:
        html = "<i>No chunks retrieved</i>"
    else:
        html = ""
        for i, c in enumerate(chunks, 1):
            preview = c.get("preview", "")[:1500]
            html += f"""
            <details>
                <summary>Chunk {i} — {c.get('drug_name','?')} ({c.get('type','?')})</summary>
                <pre style='white-space:pre-wrap'>{preview}</pre>
            </details>
            """

    return answer, drug, html


def get_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Open-Book Pharmacology Agent (Gradio UI)")
        question = gr.Textbox(label="Ask your question")
        btn = gr.Button("Submit")
        
        answer = gr.Textbox(label="Final Answer", interactive=False)
        drug = gr.Textbox(label="Extracted Drug Name", interactive=False)
        chunks = gr.HTML(label="Retrieved Context")

        btn.click(fn=ask_backend, inputs=question, outputs=[answer, drug, chunks])

    return demo
