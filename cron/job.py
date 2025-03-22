# Add these new imports at the top
from apscheduler.schedulers.background import BackgroundScheduler
import shutil

# Add this function before Firebase initialization
def check_and_delete_old_index():
    """Check and delete FAISS index directory if older than 15 minutes"""
    index_dir = "../config/faiss_index"
    try:
        if os.path.exists(index_dir):
            # Get directory modification time
            last_modified = os.path.getmtime(index_dir)
            current_time = time.time()
            
            if (current_time - last_modified) > 900:  # 15 minutes in seconds
                shutil.rmtree(index_dir)
                print(f"Deleted old FAISS index directory: {index_dir}")
                
                # Update Firebase status
                ref = db.reference('indexStatus')
                ref.set({'lastCleaned': time.ctime(), 'status': 'cleaned'})
                
    except Exception as e:
        print(f"Error cleaning FAISS index: {str(e)}")
        # Log error to Firebase
        ref = db.reference('errors')
        ref.push().set({
            'timestamp': time.ctime(),
            'error': str(e),
            'context': 'index_cleanup'
        })

# Modify the Firebase initialization block
try:
    cred = credentials.Certificate('./firebase-credentials.json')
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://secondmemoryai-default-rtdb.firebaseio.com"
    })
    print("Firebase initialized successfully")
    
    # Initialize the cleanup scheduler
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(check_and_delete_old_index, 'interval', minutes=1)
    scheduler.start()
    print("FAISS index cleanup scheduler started")

except Exception as firebase_error:
    print(f"Failed to initialize Firebase: {firebase_error}")
    raise

# Update the get_vector_store method in PDFProcessor
def get_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
    """Create vector store from documents"""
    if not documents:
        return None
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local("../config/faiss_index")  # Updated path
        # Update Firebase status
        ref = db.reference('indexStatus')
        ref.set({
            'lastUpdated': time.ctime(),
            'status': 'active',
            'documentCount': len(documents)
        })
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

# Update the search_pdfs function
def search_pdfs(query: str) -> str:
    """Search through uploaded PDFs"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index_dir = "../config/faiss_index"
        
        if not os.path.exists(index_dir):
            return "NO_PDF_AVAILABLE"
            
        pdf_db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        docs = pdf_db.similarity_search(query, k=3)
        
        if not docs:
            return "NO_RELEVANT_INFO"
            
        return "\n\n".join(f"From PDF Document:\n{doc.page_content}" for doc in docs)
            
    except Exception as e:
        return f"ERROR: {str(e)}"