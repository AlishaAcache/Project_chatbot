# [a]cache - AI Document Intelligence Platform

A professional document intelligence platform that combines Retrieval-Augmented Generation (RAG) with multi-modal AI understanding.

## Features

- **Multi-format Support**: Process PDFs, Word documents, images, code files, and more
- **AI-Powered Analysis**: Uses GPT-4 Vision for image analysis and GPT-4o-mini for text understanding
- **Semantic Search**: Advanced embedding-based document search using sentence transformers
- **Three-Mode Interface**:
  - **Ask Question**: Chat with your documents using natural language
  - **About Protocol**: Learn how [a]cache works
  - **Schedule Meeting**: Connect with the team
- **File Upload**: Drag & drop or click to upload files for instant analysis
- **Demo Questions**: Pre-built questions to help users get started

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_DEFAULT=3
INTENT_CONF_THRESHOLD=0.35
```

3. **Create required directories**:
```bash
mkdir -p docs uploads
```

4. **Add your documents** to the `docs/` folder (optional):
   - Supported formats: `.txt`, `.md`, `.pdf`, `.docx`
   - These will be indexed on startup for semantic search

## Running the Application

1. **Start the backend**:
```bash
python app.py
```
The server will start on `http://localhost:5000`

2. **Open the frontend**:
   - Open `index.html` in your web browser
   - Or serve it with a simple HTTP server:
   ```bash
   python -m http.server 8000
   ```
   Then visit `http://localhost:8000`

## How to Use

### Mode 1: Ask Question
1. Click the chat icon in the bottom-right corner
2. Select "Ask Question"
3. Either:
   - Type a question about documents in the `docs/` folder
   - Upload files using the file upload area
   - Try one of the demo questions

### Mode 2: About Protocol
- Learn how [a]cache uses RAG technology
- Understand the document processing pipeline
- Ask follow-up questions about the system

### Mode 3: Schedule Meeting
- Get contact information
- Provide meeting preferences
- Still able to ask questions about [a]cache

## File Upload Capabilities

### Supported File Types
- **Documents**: PDF, Word (.docx, .doc), Text (.txt, .md)
- **Code Files**: Python (.py), JavaScript (.js), Java (.java), C/C++ (.c, .cpp)
- **Images**: PNG, JPG, JPEG, GIF, WebP

### How It Works
1. **Upload Files**: Drag & drop or click the upload area
2. **Ask Questions**: Type your question or use the default
3. **Get AI Analysis**:
   - **Images**: Analyzed using GPT-4 Vision
   - **Documents**: Processed with text extraction and GPT understanding
   - **Code**: Analyzed for structure, purpose, and specific queries

### Example Questions for Uploaded Files
- "What is shown in this image?"
- "Summarize the key points in this document"
- "Explain what this code does"
- "What are the main topics covered?"
- "Extract the important data from this PDF"

## Demo Questions

The chatbot includes pre-built demo questions:
- "What documents are available?"
- "Summarize the key points from the documents"
- "What topics are covered in the uploaded files?"

These help users understand what they can ask.

## Architecture

### Backend (Flask)
- **RAG Pipeline**: Semantic search with sentence transformers
- **Multi-modal Processing**: GPT-4 Vision for images, text extraction for documents
- **Intent Classification**: Handles greetings, protocol questions, and document queries
- **File Upload Handler**: Secure file processing with multi-format support

### Frontend (HTML/CSS/JavaScript)
- **Modern UI**: Gradient design with [a]cache branding
- **Three-Mode Interface**: Clean separation of concerns
- **File Upload**: Drag & drop with visual feedback
- **Responsive Design**: Works on desktop and mobile

## API Endpoints

### POST /chat
Query existing documents in the `docs/` folder
```json
{
  "question": "What is the main topic of the documents?"
}
```

### POST /upload
Upload and analyze files
- Content-Type: `multipart/form-data`
- Fields:
  - `question`: Your question about the files
  - `files`: One or more files to analyze

### GET /reload
Reload document index and retrain intent classifier

### GET /doc_summary
Get two-word summaries of all indexed documents

## Configuration

Edit these variables in `app.py` or use environment variables:
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `OPENAI_MODEL`: GPT model for text generation (default: gpt-4o-mini)
- `CHUNK_SIZE`: Document chunk size (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_DEFAULT`: Number of chunks to retrieve (default: 3)

## Troubleshooting

**Issue**: "OpenAI API key not configured"
- **Solution**: Add `OPENAI_API_KEY` to your `.env` file

**Issue**: File upload fails
- **Solution**: Check file size and format. Ensure `uploads/` directory exists

**Issue**: No documents found
- **Solution**: Add documents to the `docs/` folder and restart the server

**Issue**: Vision API errors
- **Solution**: Ensure you're using a model that supports vision (gpt-4o-mini, gpt-4-vision)

## Security Notes

- Files are saved to the `uploads/` directory
- All uploads are sanitized using `secure_filename`
- API key should be kept in `.env` and never committed to version control
- Add `.env` and `uploads/` to your `.gitignore`

## Branding

The site features [a]cache branding:
- Logo: `[a]cache` with styled brackets
- Tagline: "AI-Powered Document Intelligence Platform"
- Color scheme: Purple gradient (#667eea to #764ba2)
- Professional, modern design

## License

This project is part of the [a]cache platform. Visit https://acache.co.in for more information.
