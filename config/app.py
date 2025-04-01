from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, model_validator
import uvicorn
import os
from io import BytesIO
from langchain_community.document_loaders import WebBaseLoader
import time
import validators
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
import logging
from config.RAGengine import (
    PDFProcessor,
    get_conversational_chain,
    handle_user_input
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multisource RAG API", version="2.1")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    
    @model_validator(mode='after')
    def check_question_length(cls, values):
        if len(values.question.strip()) < 3:
            raise ValueError("Question must be at least 3 characters long")
        return values

class QuestionResponse(BaseModel):
    response: str
    intermediate_steps: List[dict]

class UploadResponse(BaseModel):
    message: str
    processed_chunks: int

class WebsiteUploadRequest(BaseModel):
    urls: List[HttpUrl]
    chunk_size: Optional[int] = 512
    overlap: Optional[int] = 50

class WebsiteProcessingResponse(BaseModel):
    message: str
    processed_urls: List[str]
    stored_chunks: int

# Index Maintenance
def check_and_delete_old_index():
    """Clean up old FAISS indices"""
    for index_dir in ["./pdf_faiss_index", "./web_faiss_index"]:
        try:
            if os.path.exists(index_dir):
                last_modified = os.path.getmtime(index_dir)
                if (time.time() - last_modified) > 900:
                    shutil.rmtree(index_dir)
                    logger.info(f"Cleaned {index_dir}")
        except Exception as e:
            logger.error(f"Index cleanup error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_delete_old_index, 'interval', minutes=5)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "SecondMemory.AI API is running", "status": "ok"}
    
# Endpoints
@app.post("/upload-pdfs/", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        pdf_processor = PDFProcessor()
        pdf_contents = []
        
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(400, "Only PDF files accepted")
            content = await file.read()
            pdf_contents.append(content)
        
        raw_text = pdf_processor.get_pdf_text(pdf_contents)
        documents = pdf_processor.create_semantic_chunks(raw_text)
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise HTTPException(500, f"Embeddings initialization failed: {str(e)}")

        try:
            if os.path.exists("./pdf_faiss_index"):
                pdf_db = FAISS.load_local(
                    "./pdf_faiss_index", 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                pdf_db.add_documents(documents)
            else:
                pdf_db = FAISS.from_documents(documents, embeddings)
            
            pdf_db.save_local("./pdf_faiss_index")
        except Exception as e:
            raise HTTPException(500, f"Vector store operation failed: {str(e)}")
        
        return UploadResponse(
            message=f"Processed {len(files)} PDF(s)",
            processed_chunks=len(documents)
        )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(500, f"PDF processing failed: {str(e)}")

@app.post("/process-websites/", response_model=WebsiteProcessingResponse)
async def process_websites(request: WebsiteUploadRequest):
    try:
        pdf_processor = PDFProcessor()
        processed_urls = []
        total_chunks = 0
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise HTTPException(500, f"Embeddings initialization failed: {str(e)}")

        # Create a list to collect all documents from different URLs
        all_web_documents = []

        for url in request.urls:
            str_url = str(url)
            
            # Validate URL
            if not validators.url(str_url):
                logger.warning(f"Invalid URL: {str_url}")
                continue

            try:
                loader = WebBaseLoader(
                    [str_url],
                    requests_kwargs={
                        "timeout": 30, # Increased timeout
                        "headers": {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        },
                        "verify": False
                    }
                )
                
                docs = loader.load()
                
                # Skip if no content
                if not docs or not docs[0].page_content.strip():
                    logger.warning(f"No content found for {str_url}")
                    continue

                combined_text = "\n\n".join([d.page_content for d in docs])
                
                chunks = pdf_processor.create_semantic_chunks(
                    combined_text,
                    chunk_size=request.chunk_size,
                    overlap=request.overlap,
                    source_url=str_url  # Explicitly pass the URL as source
                )
                
                # Skip if no chunks
                if not chunks:
                    logger.warning(f"No chunks created for {str_url}")
                    continue

                # Add chunks to the collection
                all_web_documents.extend(chunks)
                total_chunks += len(chunks)
                processed_urls.append(str_url)
                
                logger.info(f"Successfully processed {str_url} with {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Unexpected error processing {str_url}: {str(e)}")
                continue

        # After processing all URLs, save to vector store
        if all_web_documents:
            try:
                if os.path.exists("./web_faiss_index"):
                    web_db = FAISS.load_local(
                        "./web_faiss_index", 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    web_db.add_documents(all_web_documents)
                else:
                    web_db = FAISS.from_documents(all_web_documents, embeddings)
                
                web_db.save_local("./web_faiss_index")
                logger.info(f"Saved {len(all_web_documents)} documents to web_faiss_index")
            except Exception as store_error:
                logger.error(f"Vector store error: {str(store_error)}")
                raise HTTPException(500, f"Failed to store web content: {str(store_error)}")

        if not processed_urls:
            raise HTTPException(400, "No valid websites could be processed")

        return WebsiteProcessingResponse(
            message=f"Processed {len(processed_urls)} websites",
            processed_urls=processed_urls,
            stored_chunks=total_chunks
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Website processing failed: {str(e)}")
        raise HTTPException(500, f"Website processing failed: {str(e)}")

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        if not request.question.strip():
            raise HTTPException(422, "Empty question provided")
            
        # Check if PDFs or websites have been processed
        pdfs_processed = os.path.exists("./pdf_faiss_index")
        websites_processed = os.path.exists("./web_faiss_index")
        
        # Get processed website URLs
        website_urls = []
        if websites_processed:
            try:
                web_index = FAISS.load_local(
                    "./web_faiss_index", 
                    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                    allow_dangerous_deserialization=True
                )
                # Extract unique source URLs from metadata
                for doc in web_index.docstore._dict.values():
                    if 'source' in doc.metadata and doc.metadata['source']:
                        website_urls.append(doc.metadata['source'])
                
                # Remove duplicates while preserving order
                seen = set()
                website_urls = [x for x in website_urls if not (x in seen or seen.add(x))]
                
                logger.info(f"Retrieved {len(website_urls)} website URLs: {website_urls[:5] if len(website_urls) > 5 else website_urls}")
            except Exception as e:
                logger.error(f"Error retrieving website URLs: {e}")
        
        try:
            # Add a timeout for agent initialization
            agent = get_conversational_chain(pdfs_processed, website_urls)
        except Exception as agent_init_error:
            logger.error(f"Agent initialization error: {str(agent_init_error)}")
            # If agent initialization fails, try without passing website URLs
            agent = get_conversational_chain(pdfs_processed, [])
        
        try:
            response = handle_user_input(request.question, agent)
        except Exception as processing_error:
            logger.error(f"Question processing error: {str(processing_error)}")
            raise HTTPException(500, f"Query processing failed: {str(processing_error)}")
        
        validated_steps = []
        for step in response.get("intermediate_steps", []):
            if not isinstance(step, dict) or "action" not in step or "observation" not in step:
                continue
            validated_steps.append({
                "action": step["action"] if isinstance(step["action"], dict) else str(step["action"]),
                "observation": str(step["observation"])
            })
            
        return QuestionResponse(
            response=response["final_response"],
            intermediate_steps=validated_steps
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Unexpected error: {str(e)}")
        
@app.get("/health")
async def health_check():
    indices = {
        "pdf": os.path.exists("./pdf_faiss_index"),
        "web": os.path.exists("./web_faiss_index")
    }
    return {"status": "healthy", "indices": indices}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
