import os
import uuid
import logging
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

# Third-party imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlite3
from contextlib import contextmanager
import pytesseract
from pdf2image import convert_from_path
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="KMRL Document Management MCP Server",
    description="Master Control Program for handling document overload at Kochi Metro Rail Limited",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = "kmrl_documents.db"

# Initialize SQLite database
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create documents table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        upload_date TIMESTAMP NOT NULL,
        content TEXT,
        department TEXT,
        project_id TEXT,
        file_path TEXT NOT NULL,
        metadata TEXT
    )
''')
conn.commit()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class DocumentType(str, Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    TENDER = "tender"
    MANUAL = "manual"
    POLICY = "policy"
    OTHER = "other"

class DocumentBase(BaseModel):
    filename: str
    doc_type: DocumentType
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    content: Optional[str] = None
    department: Optional[str] = None
    project_id: Optional[str] = None

class DocumentInDB(DocumentBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str

class DocumentResponse(DocumentBase):
    id: str
    file_path: str

    class Config:
        from_attributes = True

# Helper functions
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def classify_document(text: str) -> DocumentType:
    doc = spacy.load("en_core_web_sm")(text.lower())
    text_joined = " ".join([token.text for token in doc if not token.is_stop])
    
    # Simple keyword-based classification
    if any(word in text_joined for word in ["contract", "agreement", "mou"]):
        return DocumentType.CONTRACT
    elif any(word in text_joined for word in ["invoice", "bill", "payment"]):
        return DocumentType.INVOICE
    elif any(word in text_joined for word in ["report", "analysis", "findings"]):
        return DocumentType.REPORT
    elif any(word in text_joined for word in ["tender", "bid", "proposal"]):
        return DocumentType.TENDER
    elif any(word in text_joined for word in ["manual", "guide", "handbook"]):
        return DocumentType.MANUAL
    elif any(word in text_joined for word in ["policy", "procedure", "guideline"]):
        return DocumentType.POLICY
    return DocumentType.OTHER

# API Endpoints
@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: str = "general",
    project_id: Optional[str] = None
):
    if not file or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Generate unique filename
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        text_content = ""
        if file_ext == 'pdf':
            text_content = extract_text_from_pdf(file_path)
        else:
            # For other file types, we'll just read the text content directly
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        
        # Classify document
        doc_type = classify_document(text_content)
        
        # Create document record
        document = DocumentInDB(
            filename=file.filename,
            doc_type=doc_type,
            content=text_content,
            department=department,
            project_id=project_id,
            file_path=file_path,
            metadata={
                "original_filename": file.filename,
                "content_type": file.content_type,
                "file_size": os.path.getsize(file_path)
            }
        )
        
        # Save to database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents (id, filename, doc_type, upload_date, content, 
                                     department, project_id, file_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.id,
                document.filename,
                document.doc_type,
                document.upload_date,
                document.content,
                document.department,
                document.project_id,
                document.file_path,
                str(document.metadata)  # Store metadata as string
            ))
            conn.commit()
        
        return document
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Error processing document")

@app.get("/documents/", response_model=List[DocumentResponse])
async def list_documents(
    doc_type: Optional[DocumentType] = None,
    department: Optional[str] = None,
    project_id: Optional[str] = None,
    search: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build the query
        query_parts = []
        params = []
        
        if doc_type:
            query_parts.append("doc_type = ?")
            params.append(doc_type)
        if department:
            query_parts.append("department = ?")
            params.append(department)
        if project_id:
            query_parts.append("project_id = ?")
            params.append(project_id)
        if search:
            query_parts.append("content LIKE ?")
            params.append(f"%{search}%")
        
        where_clause = " AND ".join(query_parts) if query_parts else "1=1"
        
        cursor.execute(f"""
            SELECT * FROM documents 
            WHERE {where_clause}
            LIMIT ? OFFSET ?
        """, (*params, limit, skip))
        
        documents = cursor.fetchall()
        return [dict(row) for row in documents]

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        document = cursor.fetchone()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return dict(document)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found")
            
        conn.commit()
        return {"message": "Document deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
