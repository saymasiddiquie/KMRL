# KMRL Document Management MCP Server

This is a Master Control Program (MCP) server for managing documents at Kochi Metro Rail Limited (KMRL).

## Prerequisites

1. Python 3.8 or higher
2. MongoDB (local or remote instance)
3. Tesseract OCR (for PDF text extraction)

## Setup

1. **Install Tesseract OCR**:
   - Download and install from [UB Mannheim's Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - During installation, make sure to add Tesseract to your system PATH

2. **Clone the repository and install dependencies**:
   ```bash
   git clone <repository-url>
   cd KMRL-document-arranger
   pip install -r requirements.txt
   ```

3. **Install spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with the following content:
   ```
   MONGODB_URI=mongodb://localhost:27017/
   ```

## Running the Server

1. Start MongoDB service if it's not already running
2. Run the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
3. The server will be available at `http://127.0.0.1:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Alternative documentation: `http://127.0.0.1:8000/redoc`

## Testing the API

### Using cURL:

1. **Upload a document**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/documents/' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@path/to/your/document.pdf;type=application/pdf' \
     -F 'department=finance' \
     -F 'project_id=project123'
   ```

2. **List documents**:
   ```bash
   curl 'http://localhost:8000/documents/'
   ```

3. **Search documents**:
   ```bash
   curl 'http://localhost:8000/documents/?search=invoice&department=finance'
   ```

### Using Python:

```python
import requests

# Upload a document
url = "http://localhost:8000/documents/"
files = {'file': open('document.pdf', 'rb')}
data = {'department': 'finance', 'project_id': 'project123'}
response = requests.post(url, files=files, data=data)
print(response.json())

# Search documents
response = requests.get("http://localhost:8000/documents/", params={"search": "invoice"})
print(response.json())
```

## Project Structure

- `app.py` - Main FastAPI application
- `requirements.txt` - Python dependencies
- `uploads/` - Directory for storing uploaded files (created automatically)
- `processed/` - Directory for processed files (created automatically)
