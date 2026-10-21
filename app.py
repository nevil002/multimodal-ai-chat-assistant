from pathlib import Path
import gradio as gr
import traceback
from llm import chat_with_tools
from speech import groq_tts, to_speech_text
from rag.ingestion import ingest_file

def respond(message, history, provider, doc_ids, speak):
    text = chat_with_tools(message, history, provider, doc_ids)
    audio = groq_tts(to_speech_text(text)) if speak else None
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": text}]
    return "", history, audio

def on_upload(files, doc_ids):
    doc_ids, notes = list(doc_ids), []
    for f in files or []:
        path = Path(f)
        try:
            doc_ids.append(ingest_file(path))
            notes.append(f"Indexed {path.name}")
        except Exception as e:
            traceback.print_exc()
            notes.append(f"Could not index {path.name}: {e}")
    return list(dict.fromkeys(doc_ids)), "\n".join(notes)

with gr.Blocks(title="Multimodal AI Chat Assistant") as view:
    gr.Markdown("## Chatbot with model selector")
    doc_ids = gr.State([])                            # one list of document ids per browser session

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=450)
            with gr.Row():
                message = gr.Textbox(label="Type here", placeholder="Enter your message.....", scale=5)
                send_button = gr.Button("Send", variant="primary", scale=1)
            audio_output = gr.Audio(label="Voice Response", autoplay=True)

        with gr.Column(scale=1):
            model_config = gr.Dropdown(choices=["Gpt", "Groq"], value="Groq", label="Select the model")
            speak = gr.Checkbox(value=True, label="Speak replies")
            upload = gr.File(label="Upload documents", file_count="multiple", type="filepath")
            upload_status = gr.Markdown("No documents loaded.")
            clear_button = gr.Button("Clear documents")

    chat_inputs = [message, chatbot, model_config, doc_ids, speak]
    chat_outputs = [message, chatbot, audio_output]
    send_button.click(respond, inputs=chat_inputs, outputs=chat_outputs)
    message.submit(respond, inputs=chat_inputs, outputs=chat_outputs)       # Enter key
    upload.upload(on_upload, inputs=[upload, doc_ids], outputs=[doc_ids, upload_status])
    clear_button.click(lambda: ([], "No documents loaded."), inputs=None, outputs=[doc_ids, upload_status])

view.launch()