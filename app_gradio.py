import gradio as gr
from implementation.answer import answer_question

def add_user_message(query, history):
    history.append({"role": "user", "content": query})
    return "", history

def process_query(history):
    # Extract the user's query from the history
    query = history[-1]["content"]
    # Pass history without the latest query to answer_question
    query_history = history[:-1]
    
    # Call the backend function
    response, docs = answer_question(query, query_history)
    
    # Update the chat history with the assistant's response
    history.append({"role": "assistant", "content": response})
    
    # Format the retrieved documents into a single Markdown string
    markdown_output = ""
    for i, doc in enumerate(docs):
        # We access doc.page_content since answer.py returns Langchain Document objects
        markdown_output += f"# **Retrieval_{i + 1}**\n{doc.page_content}\n\n"
        
    # Return the updated history and the markdown output
    return history, markdown_output

with gr.Blocks(title="Insurelm chat") as demo:
    gr.Markdown("# Insurelm chat")
    
    with gr.Row():
        # Left Column: Chatbot and Textbox
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat with Insurelm AI Agent", type="messages", height=600)
            user_input = gr.Textbox(label="Enter your query:", placeholder="Type a message and press Enter...")
            submit_btn = gr.Button("Submit")
            
        # Right Column: Markdown viewer for retrieval chunks
        with gr.Column():
            gr.Markdown("## Langchain Retrieval Chunks")
            retrieval_display = gr.Markdown(value="*Retrieved chunks will appear here after you ask a question.*", container=True, height=600)

    # Connect the submit action for both "Enter" on textbox and button click
    user_input.submit(
        fn=add_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    ).then(
        fn=process_query,
        inputs=[chatbot],
        outputs=[chatbot, retrieval_display]
    )
    submit_btn.click(
        fn=add_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    ).then(
        fn=process_query,
        inputs=[chatbot],
        outputs=[chatbot, retrieval_display]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
