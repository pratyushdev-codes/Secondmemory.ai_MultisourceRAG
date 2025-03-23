from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
from io import BytesIO
import time
import validators
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
## Chaning this --> lets see if it works
from .RAGengine import (
    PDFProcessor,
    get_conversational_chain,
    handle_user_input,
    db
)


app = FastAPI(title="Multisource RAG API", version="2.0")

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
    duplicates_skipped: int

# Index Maintenance
def check_and_delete_old_index():
    """Clean up old FAISS indices"""
    for index_dir in ["./pdf_faiss_index", "./web_faiss_index"]:
        try:
            if os.path.exists(index_dir):
                last_modified = os.path.getmtime(index_dir)
                if (time.time() - last_modified) > 900:  # 15 minutes
                    shutil.rmtree(index_dir)
                    print(f"Cleaned {index_dir}")
                    db.reference('indexStatus').push().set({
                        'indexType': os.path.basename(index_dir),
                        'lastCleaned': time.ctime()
                    })
        except Exception as e:
            print(f"Index cleanup error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_delete_old_index, 'interval', minutes=5)
    scheduler.start()

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
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if os.path.exists("./pdf_faiss_index"):
            pdf_db = FAISS.load_local("./pdf_faiss_index", embeddings)
            pdf_db.add_documents(documents)
        else:
            pdf_db = FAISS.from_documents(documents, embeddings)
        
        pdf_db.save_local("./pdf_faiss_index")
        
        return UploadResponse(
            message=f"Processed {len(files)} PDF(s)",
            processed_chunks=len(documents)
        )  # <-- This closing parenthesis was missing
            
    except Exception as e:
        raise HTTPException(500, f"PDF processing failed: {str(e)}")

@app.post("/process-websites/", response_model=WebsiteProcessingResponse)
async def process_websites(request: WebsiteUploadRequest):
    try:
        pdf_processor = PDFProcessor()
        processed_urls = []
        total_chunks = 0
        duplicates = 0
        
        existing_data = db.child("websiteData").get().val() or {}
        existing_urls = {v['url'] for v in existing_data.values() if 'url' in v}
        
        for url in request.urls:
            str_url = str(url)
            if str_url in existing_urls:
                duplicates += 1
                continue
                
            try:
                loader = WebBaseLoader([str_url])
                docs = loader.load()
                combined_text = "\n\n".join([d.page_content for d in docs])
                
                chunks = pdf_processor.create_semantic_chunks(
                    combined_text,
                    chunk_size=request.chunk_size,
                    overlap=request.overlap
                )
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                if os.path.exists("./web_faiss_index"):
                    web_db = FAISS.load_local("./web_faiss_index", embeddings)
                    web_db.add_documents(chunks)
                else:
                    web_db = FAISS.from_documents(chunks, embeddings)
                
                web_db.save_local("./web_faiss_index")
                total_chunks += len(chunks)
                
                db.child("websiteData").push({
                    "url": str_url,
                    "processed_at": time.time(),
                    "chunks": len(chunks)
                })
                processed_urls.append(str_url)
                
            except Exception as e:
                print(f"Failed {str_url}: {str(e)}")
                continue

        return WebsiteProcessingResponse(
            message=f"Processed {len(processed_urls)} websites",
            processed_urls=processed_urls,
            stored_chunks=total_chunks,
            duplicates_skipped=duplicates
        )
        
    except Exception as e:
        raise HTTPException(500, f"Website processing failed: {str(e)}")

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        agent = get_conversational_chain()
        response = handle_user_input(request.question, agent)
        return QuestionResponse(**response)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        db.child("healthCheck").set(time.time())
        indices = {
            "pdf": os.path.exists("./pdf_faiss_index"),
            "web": os.path.exists("./web_faiss_index")
        }
        return {"status": "healthy", "indices": indices}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
