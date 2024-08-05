
from flask import Flask, request, render_template, redirect, url_for
from unstructured.partition.pdf import partition_pdf
import os
import uuid
import base64
import pytesseract
from PIL import Image
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from IPython.display import display, HTML
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from pinecone import Pinecone, ServerlessSpec
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls..."
os.environ["LANGCHAIN_PROJECT"] = "langchain-multimodal-pdf"
os.environ["GOOGLE_API_KEY"] = "AI..."
os.environ["GROQ_API_KEY"] = "gsk_..."

# Initialize API keys
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Pinecone client
pinecone_api_key = "your_pinecone_api_key"

pinecone_index_name="multi-modal-rag"
pc=Pinecone(api_key=pinecone_api_key)

if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
pinecone_index=pc.Index(pinecone_index_name)


# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to summarize images
def image_summarize(prompt, img_base64):
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ])
    ])
    return msg.content

# Function to split documents by type
def split_image_text_types(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            base64.b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

# Create Flask application
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

pdf_elements = None
retriever = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global pdf_elements, retriever
    response = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            pdf_elements, retriever = process_pdf(filepath)
            return render_template('upload.html', pdf_uploaded=True)
    return render_template('upload.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global retriever
    if request.method == 'POST':
        question = request.form['question']
        response = generate_response(question, retriever)
        return render_template('upload.html', response=response, pdf_uploaded=True)
    return redirect(url_for('upload_file'))

def process_pdf(pdf_path):
    # Extract elements from PDF
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir="extracted_data"
    )

    # Categorize extracted elements
    Table, Text, Image = [], [], []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            Table.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            Text.append(str(element))
        elif "unstructured.documents.elements.Image" in str(type(element)):
            Image.append(str(element))

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    # Apply to text
    text_summaries = Text

    # Apply to tables
    table_summaries = summarize_chain.batch(Table, {"max_concurrency": 5})

    # Image summaries
    path = "extracted_data"
    img_base64_list = []
    image_summaries = []
    prompt = "Describe the image in detail. Be specific about graphs, such as bar plots."

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith('.jpg') and img_file.startswith('figure'):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(prompt, base64_image))

    # Add to vectorstore
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_function)
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in Text]
    summary_texts = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, Text)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in Table]
    summary_tables = [Document(page_content=s, metadata={id_key: table_ids[i]}) for i, s in enumerate(table_summaries)]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, Table)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
    summary_img = [Document(page_content=s, metadata={id_key: img_ids[i]}) for i, s in enumerate(image_summaries)]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, img_base64_list)))

    # Upsert documents into Pinecone
    pinecone_docs = summary_texts + summary_tables + summary_img
    pinecone_vectors = [
        {"id": doc.metadata[id_key], "values": embedding_function.embed_query(doc.page_content), "metadata": {"content": doc.page_content}}
        for doc in pinecone_docs
    ]
    pinecone_index.upsert(vectors=pinecone_vectors, namespace="ns1")

    return raw_pdf_elements, retriever

def generate_response(question, retriever):
    # Check retrieval
    docs = retriever.get_relevant_documents(question)
    docs_by_type = split_image_text_types(docs)

    def plt_img_base64(img_base64):
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        display(HTML(image_html))

    plt_img_base64(docs_by_type["images"][0])

    # RAG pipeline
    def prompt_func(dict):
        format_texts = "\n".join(dict["context"]["texts"])
        return [
            HumanMessage(content=[
                {"type": "text", "text": f"""Answer the question based only on the following context, which can include text, tables, and the below image:
    Question: {dict["question"]}

    Text and tables:
    {format_texts}
    """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dict['context']['images'][0]}"}},
            ])
        ]

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chain = (
        {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
        | RunnableLambda(prompt_func)
        | model
        | StrOutputParser()
    )

    # Generate response
    response = chain.invoke(question)
    return response

if __name__ == "__main__":
    app.run(debug=True)
