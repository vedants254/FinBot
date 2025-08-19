# FinBoT: Financial Analysis Made Easy! 
![FinBoT Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![LangChain](https://img.shields.io/badge/LangChain-Latest-yellow)

 
FinBoT- Finance Bot is your intelligent financial analysis assistant built to help you understand complex accounting reports through natural language interaction. Powered by Retrieval-Augmented Generation (RAG) and LLMs, it extracts, interprets, and explains financial data with precision.

## Just upload your reports. FinBoT does the rest:
1. Compare financials across companies—shareholder equity, revenue, expenses, and more
2. Detect red flags such as lawsuits, audit opinions, or going concern issues
3. Summarize performance metrics, trends, and ratios across time periods
4. Export structured insights to Excel or dashboards for deeper use
5. Interact with a chat UI that maintains memory and context

Whether you're a CFO, investor, analyst, or student, FinBoT makes financial reports readable, insightful, and actionable. 

**No spreadsheets. No manual analysis. Just smart, explainable finance.**


## Features

- **PDF Document Processing**: Upload and analyze financial reports, annual statements, and other PDF documents
- **Intelligent Q&A**: Ask questions about your uploaded documents and get detailed financial insights
- **Table Extraction**: Automatically extracts both text and tabular data from PDFs
- **Conversational Memory**: Maintains context throughout your conversation
- **Real-time Processing**: Fast document chunking and embedding generation
- **Professional UI**: Clean, modern interface with chat-style interactions

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (Llama3.1)
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **Vector Store**: ChromaDB
- **PDF Processing**: PDFPlumber
- **Framework**: LangChain
- **Memory**: Conversation Buffer Memory

## Project Structure

```
finbot/
├── __pycache__/          # Python cache files
├── data/                 # Data storage directory
├── demo/                 # Demo files and examples
├── myenv/               # Virtual environment
├── .env                 # Environment variables
├── app.py               # Main Streamlit application
├── html_templates.py    # CSS and HTML templates for chat UI
├── main.py              # Core application logic
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/finbot.git
   cd finbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_PROJECT=your_project_name
   HF_TOKEN=your_huggingface_token_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   ```

5. **Install Ollama and download Llama3.1 8B **
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   ollama pull llama3.1:8b
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:8501`

3. **Upload documents**
   - Use the sidebar to upload PDF financial reports
   - Click "Process" to analyze the documents
   - Wait for the "Documents processed successfully!" message

4. **Start asking questions**
   - Type your financial analysis questions in the text input
   - Get detailed, context-aware responses based on your uploaded documents

## Example Queries

- "Compare the shareholder's equity of company XYZ and American Express"
- "What is the approximate principal repayments amount?"
- "Analyze the revenue trends over the past 3 years"
- "What are the key financial ratios mentioned in the report?"
- "Summarize the company's debt structure"

## Dependencies

```txt
langchain_core
langchain_community 
langsmith
streamlit 
faiss-cpu
python-dotenv
langserve
PyPDF2
chromadb
pdfplumber
transformers
fastembed
```

## Key Components

### Document Processing
- **PDF Text Extraction**: Extracts text content from uploaded PDFs
- **Table Processing**: Converts tables to text format for analysis
- **Text Chunking**: Splits documents into manageable chunks with overlap
- **Embedding Generation**: Creates vector embeddings for semantic search

### AI Pipeline
- **Retrieval Chain**: Uses ConversationalRetrievalChain for context-aware responses
- **Memory Management**: Maintains conversation history for coherent interactions
- **Custom Prompts**: Specialized prompts for financial analysis tasks

### User Interface
- **Chat Interface**: Professional chat-style UI with user and bot avatars
- **File Upload**: Drag-and-drop file upload with progress indicators
- **Real-time Processing**: Live feedback during document processing

## Environment Setup

The application uses several environment variables for configuration:

- `LANGCHAIN_TRACING_V2`: Enables LangChain tracing
- `LANGCHAIN_ENDPOINT`: LangChain API endpoint
- `LANGCHAIN_PROJECT`: Project name for tracking
- `HF_TOKEN`: HuggingFace API token
- `LANGCHAIN_API_KEY`: LangChain API key

## Performance

- **Document Processing**: Optimized chunking with 1000 character chunks and 200 character overlap
- **Embedding Model**: Efficient BAAI/bge-small-en-v1.5 model for fast similarity search
- **Memory Management**: ConversationBufferMemory for maintaining context
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.