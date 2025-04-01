from typing import List
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List, Optional
from io import BytesIO
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["USER_AGENT"] = "secondmemory.ai/1.0 (birole.pratyush@gmail.com)"

class PDFProcessor:
    def get_pdf_text(self, pdf_contents: List[bytes]) -> str:
        """Extract text from PDF bytes"""
        text = ""
        for content in pdf_contents:
            pdf_reader = PdfReader(BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
        
    def create_semantic_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50, source_url: str = None) -> List[Document]:
        """Create semantic chunks from text with proper source tracking"""
        initial_splits = text.split('\n\n')
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        documents = []
        for i, section in enumerate(initial_splits):
            if not section.strip():
                continue
                
            try:
                metadata = {
                    "chunk_id": f"section_{i}", 
                    "chunk_type": "full_section"
                }
                
                # Add source URL if provided and ensure it's a string
                if source_url:
                    metadata["source"] = str(source_url)
                    
                if len(section) > chunk_size:
                    chunks = splitter.split_text(section)
                    for j, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = f"section_{i}_chunk_{j}"
                        documents.append(Document(
                            page_content=chunk,
                            metadata=chunk_metadata
                        ))
                else:
                    documents.append(Document(
                        page_content=section,
                        metadata=metadata
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing section {i}: {str(e)}")
                continue
                
        return documents
        
    def get_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """Create vector store from documents"""
        if not documents:
            return None
            
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None

def search_pdfs(query: str) -> str:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("./pdf_faiss_index"):
            return "NO_PDF_AVAILABLE"
            
        pdf_db = FAISS.load_local(
            "./pdf_faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = pdf_db.similarity_search(query, k=3)
        
        if not docs:
            return "NO_RELEVANT_INFO"
            
        return "\n\n".join(f"From PDF Document:\n{doc.page_content}" for doc in docs)
            
    except Exception as e:
        logger.error(f"PDF search error: {str(e)}")
        return f"ERROR: {str(e)}"

def search_websites(query: str) -> str:
    """Search within web content that's been processed and indexed"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("./web_faiss_index"):
            return "NO_WEB_CONTENT_AVAILABLE"
            
        web_db = FAISS.load_local(
            "./web_faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = web_db.similarity_search(query, k=3)
        
        if not docs:
            return "NO_RELEVANT_INFO"
            
        results = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown Source')
            results.append(f"From Web Document ({source}):\n{doc.page_content}")
            
        return "\n\n".join(results)
            
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return f"ERROR: {str(e)}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_tools(pdfs_processed: bool = False, websites: List[str] = []):
    """Create tools for the agent"""
    tools = []
    
    # Add Wikipedia tool
    wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
    tools.append(Tool(
        name="wikipedia",
        func=wiki_tool.run,
        description="Search Wikipedia for general knowledge information."
    ))
    
    # Add Arxiv tool
    arxiv = ArxivAPIWrapper(top_k_results=2, sort_by="relevancy")
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv)
    tools.append(Tool(
        name="arxiv",
        func=arxiv_tool.run,
        description="Search Arxiv for scientific papers and research."
    ))
    
    # Add Web Search tool if web index exists or websites provided
    if os.path.exists("./web_faiss_index") or websites:
        web_tool = Tool(
            name="web_search",
            func=search_websites,
            description="Search within processed web content for information"
        )
        tools.append(web_tool)
    
    # Add news tool 
    try:
        news_urls = ["https://www.reuters.com/world/"]
        
        loader = WebBaseLoader(
            news_urls,
            requests_kwargs={
                "timeout": 10000,
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                "verify": False
            }
        )
        
        loaded_docs = loader.load()
        
        # Combine text with proper metadata
        news_documents = []
        pdf_processor = PDFProcessor()
        
        for doc in loaded_docs:
            # Limit content to avoid processing too much text
            limited_content = doc.page_content[:2000]
            chunks = pdf_processor.create_semantic_chunks(
                limited_content,
                chunk_size=300,
                overlap=30,
                source_url=doc.metadata.get("source", "news")
            )
            news_documents.extend(chunks)
        
        if news_documents:
            news_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            news_vectordb = FAISS.from_documents(news_documents, news_embeddings)
            
            news_tool = create_retriever_tool(
                news_vectordb.as_retriever(search_kwargs={"k": 3}),
                "news_search",
                "Search for recent global news and updates from Reuters."
            )
            tools.append(news_tool)
        else:
            logger.warning("No news documents could be processed")
    
    except Exception as e:
        logger.error(f"News tool creation failed: {e}")
    
    # Add PDF search tool if PDFs processed
    if pdfs_processed:
        pdf_tool = Tool(
            name="pdf_search",
            func=search_pdfs,
            description="Search within uploaded PDF documents"
        )
        tools.append(pdf_tool)
    
    return tools

def get_conversational_chain(pdfs_processed: bool = False, websites: List[str] = []):
    """Create the conversational chain"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )
    
    tools = create_tools(pdfs_processed, websites)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output" 
    )
    
    system_message = """You are a helpful AI assistant that can search through multiple sources including PDFs, Wikipedia, Arxiv, and web documentation.
    When using the tools:
    - If you receive 'NO_PDF_AVAILABLE', inform the user that no PDFs have been uploaded yet.
    - If you receive 'NO_WEB_CONTENT_AVAILABLE', inform the user that no web URLs have been processed yet.
    - If you receive 'NO_RELEVANT_INFO', inform the user that no relevant information was found.
    - Always provide detailed answers by combining information from multiple sources when appropriate."""
    
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        system_message=system_message,
        return_intermediate_steps=True
    )

def handle_user_input(user_question: str, agent_executor: AgentExecutor) -> dict:
    """Process user input and return response with intermediate steps"""
    try:
        response = agent_executor.invoke({
            "input": user_question,
        })
        
        # Extract intermediate steps
        steps = []
        if response.get("intermediate_steps"):
            for step in response["intermediate_steps"]:
                # Extract action details
                action = step[0]
                observation = step[1]
                
                # More robust action details extraction
                action_details = {
                    "tool": getattr(action, 'tool', str(action)),
                    "tool_input": getattr(action, 'tool_input', ''),
                    "log": getattr(action, 'log', '')
                }
                
                # Add step to list
                steps.append({
                    "action": action_details,
                    "observation": str(observation)
                })
        
        return {
            "final_response": response.get("output", "No response generated"),
            "intermediate_steps": steps
        }
    except Exception as e:
        import traceback
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Specific handling for quota exhaustion
        if "ResourceExhausted" in str(e) or "429" in str(e):
            return {
                "final_response": "I'm temporarily unable to process your request due to API quota limits. Please try again later.",
                "intermediate_steps": []
            }
        
        return {
            "final_response": f"An error occurred: {str(e)}",
            "intermediate_steps": []
        }
