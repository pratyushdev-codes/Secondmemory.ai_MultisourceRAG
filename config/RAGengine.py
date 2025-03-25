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
from typing import List
from pyrebase import pyrebase  # Change this line
from typing import List, Optional
from io import BytesIO
import time
from langchain_community.document_loaders import WebBaseLoader
# from firebase import Firebase
import pyrebase
os.environ["USER_AGENT"] = "secondmemory.ai/1.0 (birole.pratyush@gmail.com)"
# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



## Firebase configuration



# Firebase configuration
# Initialize Firebase Admin SDK

# Firebase configuration (keep this as in original code)
config = {
    "apiKey": "AIzaSyAzvf9coRhksEk8Zhcwuw4EVKjJgcQovgY",
    "authDomain": "secondmemoryai.firebaseapp.com",
    "databaseURL": "https://secondmemoryai-default-rtdb.firebaseio.com",
    "storageBucket": "secondmemoryai.appspot.com"
}

# firebase = pyrebase.initialize_app(config)
# db = firebase.database()

try:
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
except Exception as e:
    print(f"Firebase initialization error: {str(e)}")
    # Provide a fallback db that won't crash your app
    class DummyDB:
        def child(self, path):
            return self
        def get(self):
            return DummyResponse()
        def push(self):
            return self
        def set(self, value):
            pass
            
    class DummyResponse:
        def val(self):
            return {}
            
    db = DummyDB()
    

class PDFProcessor:
    def __init__(self):
        self.fetched_web_urls = self._get_web_urls()

    def _get_web_urls(self) -> List[str]:
        """Fetch website URLs from Firebase Realtime Database using python-firebase"""
        try:
            # Get website data using the third-party Firebase library
            website_data = db.child("websiteData").get().val()
            
            if not website_data:
                return []

            # Extract URLs from the nested Firebase structure
            return [
                entry.get('url') 
                for entry in website_data.values() 
                if entry and 'url' in entry
            ]
        except Exception as e:
            print(f"Error fetching web URLs from Firebase: {e}")
            return []
            
    def get_pdf_text(self, pdf_contents: List[bytes]) -> str:
        """Extract text from PDF bytes"""
        text = ""
        for content in pdf_contents:
            pdf_reader = PdfReader(BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
        
    def create_semantic_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Document]:
        """Create semantic chunks from text"""
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
                if len(section) > chunk_size:
                    chunks = splitter.split_text(section)
                    documents.extend([
                        Document(
                            page_content=chunk,
                            metadata={"source": f"section_{i}", "chunk_type": "split_chunk"}
                        ) for chunk in chunks if chunk.strip()
                    ])
                else:
                    documents.append(Document(
                        page_content=section,
                        metadata={"source": f"section_{i}", "chunk_type": "full_section"}
                    ))
                    
            except Exception as e:
                print(f"Error processing section {i}: {str(e)}")
                continue
                
            time.sleep(0.1)
            
        return documents
        
    def get_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """Create vector store from documents"""
        if not documents:
            return None
            
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None

def create_tools(pdf_processor: PDFProcessor):
    """Create tools for the agent"""
    tools = []
    
    # Add Wikipedia tool
    wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
    tools.append(wiki_tool)
    
    # Add Arxiv tool
    arxiv = ArxivAPIWrapper(top_k_results=2, sort_by="relevancy")
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv)
    tools.append(arxiv_tool)
    
    # Add web search tool if URLs available
    if pdf_processor.fetched_web_urls:
        try:
            loader = WebBaseLoader(
                pdf_processor.fetched_web_urls,
                verify_ssl=False,
                requests_kwargs={
                    "timeout": 10,
                    "headers": {"User-Agent": os.environ["USER_AGENT"]}
                }
            )
            documents = pdf_processor.create_semantic_chunks("\n\n".join(doc.page_content for doc in loader.load()))
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = FAISS.from_documents(documents, embeddings)
            web_tool = create_retriever_tool(
                vectordb.as_retriever(search_kwargs={"k": 3}),
                "web_search",
                "Search web documentation for technical information."
            )
            tools.append(web_tool)
        except Exception as e:
            print(f"Web tool creation failed: {e}")
    
    # Add News Tool
    try:
        news_urls = [
            "https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN:en",
        ]
        
        news_loader = WebBaseLoader(
            news_urls,
            verify_ssl=False,
            requests_kwargs={
                "timeout": 10,
                "headers": {"User-Agent": os.environ["USER_AGENT"]}
            }
        )
        news_documents = pdf_processor.create_semantic_chunks("\n\n".join(doc.page_content for doc in news_loader.load()))
        
        news_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        news_vectordb = FAISS.from_documents(news_documents, news_embeddings)
        news_retriever = news_vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        news_tool = create_retriever_tool(
            news_retriever,
            "news_search",
            "Search for recent news, top news, trends, and real-time updates. Use this for current events and real-time knowledge."
        )
        tools.append(news_tool)
    except Exception as e:
        print(f"News tool creation failed: {e}")
            
    # Add PDF search tool
    pdf_tool = Tool(
        name="pdf_search",
        func=search_pdfs,
        description="Search within uploaded PDF documents"
    )
    tools.append(pdf_tool)
    
    return tools

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
        return f"ERROR: {str(e)}"

def get_conversational_chain():
    """Create the conversational chain"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )
    
    pdf_processor = PDFProcessor()
    tools = create_tools(pdf_processor)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    system_message = """You are a helpful AI assistant that can search through multiple sources including PDFs, Wikipedia, Arxiv, and web documentation.
    When using the pdf_search tool:
    - If you receive 'NO_PDF_AVAILABLE', inform the user that no PDFs have been uploaded yet.
    - If you receive 'NO_RELEVANT_INFO', inform the user that no relevant information was found.
    - If you receive actual content, incorporate it into your response with citations.
    
    Always provide detailed answers by combining information from multiple sources when appropriate."""
    
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
        return_intermediate_steps=True  # Add this line
    )

# Then update the handle_user_input function:
# def handle_user_input(user_question: str, agent_executor: AgentExecutor) -> dict:
#     """Process user input and return response with intermediate steps"""
#     try:
#         response = agent_executor.invoke({
#             "input": user_question,
#         })
        
#         # Extract intermediate steps
#         steps = []
#         if "intermediate_steps" in response:
#             for step in response["intermediate_steps"]:
#                 # Extract action details
#                 action = step[0]
#                 observation = step[1]
                
#                 # Format action details
#                 action_details = {
#                     "tool": action.tool if hasattr(action, 'tool') else str(action),
#                     "tool_input": action.tool_input if hasattr(action, 'tool_input') else "",
#                     "log": action.log if hasattr(action, 'log') else ""
#                 }
                
#                 # Add step to list
#                 steps.append({
#                     "action": action_details,
#                     "observation": str(observation)
#                 })
        
#         return {
#             "final_response": response["output"],
#             "intermediate_steps": steps
#         }
#     except Exception as e:
#         raise Exception(f"Error processing question: {str(e)}") 
def handle_user_input(user_question: str, agent_executor: AgentExecutor) -> dict:
    """Process user input and return response with intermediate steps"""
    try:
        response = agent_executor.invoke({
            "input": user_question,
        })
        
        # Extract intermediate steps
        steps = []
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                # Extract action details
                action = step[0]
                observation = step[1]
                
                # Format action details
                action_details = {
                    "tool": action.tool if hasattr(action, 'tool') else str(action),
                    "tool_input": action.tool_input if hasattr(action, 'tool_input') else "",
                    "log": action.log if hasattr(action, 'log') else ""
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
        print(f"Error processing question: {str(e)}")
        return {
            "final_response": f"An error occurred: {str(e)}",
            "intermediate_steps": []
        }
