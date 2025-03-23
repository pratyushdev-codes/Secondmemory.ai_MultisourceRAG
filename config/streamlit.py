# app.py
import streamlit as st
from io import BytesIO
from typing import List
from RAGengine import PDFProcessor, get_conversational_chain, handle_user_input

def main():
    st.set_page_config(
        page_title="SecondMemory.AI",
        page_icon="🧠",
        layout="centered"
    )
    
    st.title("🧠 SecondMemory.AI - Multi-Source RAG System")
    st.caption("Powered by Gemini, FAISS, and LangChain")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "show_steps" not in st.session_state:
        st.session_state.show_steps = False

    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # PDF Upload
        uploaded_pdfs = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        # Website URLs
        website_urls = st.text_area(
            "Enter website URLs (one per line)",
            height=100
        )
        
        # Processing button
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                pdf_contents = []
                if uploaded_pdfs:
                    pdf_contents = [pdf.getvalue() for pdf in uploaded_pdfs]
                
                # Process PDFs
                pdf_processor = PDFProcessor()
                if pdf_contents:
                    raw_text = pdf_processor.get_pdf_text(pdf_contents)
                    documents = pdf_processor.create_semantic_chunks(raw_text)
                    pdf_processor.get_vector_store(documents)
                
                # Save website URLs to Firebase
                if website_urls:
                    urls = [url.strip() for url in website_urls.split('\n') if url.strip()]
                    try:
                        ref = db.reference('websiteData')
                        for url in urls:
                            ref.push({'url': url})
                    except Exception as e:
                        st.error(f"Error saving URLs: {e}")
                
                # Initialize agent after processing
                try:
                    st.session_state.agent_executor = get_conversational_chain()
                    st.session_state.processed = True
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error initializing agent: {e}")

        # Toggle for intermediate steps
        st.session_state.show_steps = st.checkbox("Show intermediate steps")

    # Main chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if st.session_state.show_steps and message.get("steps"):
                with st.expander("Intermediate Steps"):
                    for step in message["steps"]:
                        st.json(step)

    # User input
    if prompt := st.chat_input("Ask your question..."):
        if not st.session_state.processed:
            st.warning("Please process documents first!")
            st.stop()
            
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        try:
            with st.spinner("Analyzing..."):
                response = handle_user_input(
                    prompt,
                    st.session_state.agent_executor
                )
                
            # Display response
            with st.chat_message("assistant"):
                st.markdown(response["final_response"])
                if st.session_state.show_steps:
                    with st.expander("Intermediate Steps"):
                        st.json(response["intermediate_steps"])
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["final_response"],
                "steps": response["intermediate_steps"] if st.session_state.show_steps else None
            })
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()