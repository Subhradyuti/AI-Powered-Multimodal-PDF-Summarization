# Multimodal PDF Summarization and Q&A System

![PDF](https://github.com/user-attachments/assets/6c067344-1c89-4c9a-ac8b-bcc53db6b13d)

## Overview
This project demonstrates a sophisticated system for processing PDF documents by extracting and summarizing their textual, tabular, and image content. The summarized information is then indexed in a vector database to enable effective question-answering capabilities. The system leverages multiple advanced tools and methods to achieve this, providing a comprehensive solution for handling and interacting with multimodal content within PDFs.

## Tools and Libraries Used
- **Flask**: A lightweight web framework used to create the web interface for uploading PDFs and asking questions.
- **Unstructured**: A library used to partition PDF documents into different elements such as text, tables, and images.
- **Pytesseract**: An optical character recognition (OCR) tool used to extract text from images within the PDFs.
- **PIL (Python Imaging Library)**: Used for handling image files.
- **LangChain**: A framework for building applications with language models, used here for creating prompts and managing the summarization and Q&A process.
- **Chroma**: A vector database used to store and retrieve summarized content.
- **Pinecone**: A scalable vector search engine used for storing and retrieving vectorized representations of the summarized content.
- **Google Generative AI**: Used for generating embeddings and summarizations for text and images.
- **Groq**: An API used to generate summaries from textual content.
- **UUID**: For generating unique identifiers for documents and images.
- **Base64**: For encoding images to be used within prompts and web pages.
![image](https://github.com/user-attachments/assets/6628d59b-9968-4ba2-8327-9eb0a211514b)


## How It Works

### PDF Upload and Processing:
1. The user uploads a PDF file through the web interface created using Flask.
2. The PDF is saved in a specified directory, and its content is partitioned into different elements (text, tables, images) using the Unstructured library.

### Content Summarization:
- **Text and Tables**: The text and table elements are processed using a summarization chain. The chain uses a prompt template to structure the input and the Groq API to generate concise summaries.
- **Images**: Images are processed by encoding them to Base64 and then generating a descriptive summary using the Google Generative AI model.

### Storing Summarized Content:
1. The summarized text, tables, and images are embedded into vector representations using the Google Generative AI embeddings.
2. These embeddings, along with the original content, are stored in a Chroma vector store and indexed in a Pinecone vector database for efficient retrieval.

### Question-Answering (Q&A):
1. When a user submits a question through the web interface, the retriever fetches relevant documents based on the question.
2. The retrieved documents are categorized into text, tables, and images.
3. A structured prompt is created using the question and the retrieved content, including images in Base64 format.
4. The prompt is passed to the Google Generative AI model to generate a response, which is then displayed to the user.

## Key Functions
- `encode_image(image_path)`: Encodes an image to Base64 format.
- `image_summarize(prompt, img_base64)`: Generates a summary for an image using the Google Generative AI model.
- `split_image_text_types(docs)`: Splits retrieved documents into text and images.
- `process_pdf(pdf_path)`: Processes the uploaded PDF to extract and summarize content, then stores it in vector databases.
- `generate_response(question, retriever)`: Generates a response to a user question by retrieving and processing relevant documents.

## Installation
1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/multimodal-pdf-summarization-qa.git
  cd multimodal-pdf-summarization-qa

2. Create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:
  ```bash
  pip install -r requirements.txt

4. Set up the environment variables in a .env file:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
    LANGCHAIN_API_KEY=your_langchain_api_key
    LANGCHAIN_PROJECT=langchain-multimodal-pdf
    GOOGLE_API_KEY=your_google_api_key
    GROQ_API_KEY=your_groq_api_key
    Install Tesseract OCR and set the path in your environment variables:

5. Download and install Tesseract OCR from here.
Add the Tesseract installation path to your environment variables:
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

## Usage
Run the Flask application:
python app.py
Open your browser and go to http://127.0.0.1:5000/ to upload a PDF and interact with the system.

## Contributing
Fork the repository.
Create a new branch: git checkout -b feature-branch
Make your changes and commit them: git commit -m 'Add new feature'
Push to the branch: git push origin feature-branch
Submit a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.


## Conclusion
This project showcases an advanced system for extracting, summarizing, and interacting with multimodal content in PDF documents. By integrating multiple AI models and scalable vector databases, it provides an efficient and user-friendly solution for handling complex documents and facilitating effective question-answering capabilities.
