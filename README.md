# Healthcare FlaskAPP 

## Project Overview
This repository contains an AI-based question-answering system tailored for healthcare-related queries, leveraging LangChain, Chroma, and OpenAI embeddings. The AI provides concise and reliable responses related to healthcare or Omnidoc-specific questions while avoiding any medical diagnosis or treatment plans. The system uses document retrieval and large language models to deliver contextually relevant responses, incorporating CORS-enabled Flask API for communication.

## Requirements
- Python 3.x
- OpenAI API Key
- GROQ API Key
- FitZ (PyMuPDF)
- Flask
- LangChain and related dependencies
- Chroma

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Setting Up the Environment
1. Create a .env file in the project root directory and include the following environment variables:
```bash
OPENAI_API_KEY=<your_openai_api_key>
GROQ_API_KEY=<your_groq_api_key>
```
2. Place your PDF documents in the data/pdfs/ directory to be processed for embeddings.

## Usage
### Run the Server
Ensure all dependencies are installed and environment variables are set.
Start the Flask server:
```bash
python server.py
```

## Example API Request
- Endpoint: /question
- Method: POST
- Content-Type: application/json
- Body:
```bash
{
  "question": "When was Omnidoc founded?"
}
```

- Response:
```bash
{
  "message": "Question received",
  "answer": "Omnidoc was founded in 2020."
}
```


## Future Improvements
- Add additional support for different languages in response generation.
- Improve document retrieval accuracy by fine-tuning the AI model.
- Expand PDF processing for more complex document structures.
