import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore

# एनवायरनमेंट वेरिएबल्स लोड करें
load_dotenv()

app = Flask(__name__)

# 1. अपनी मेडिकल टेक्स्ट फाइल को लोड करें
loader = TextLoader("data/medical_data.txt", encoding="utf-8")
documents = loader.load()

# 2. टेक्स्ट को छोटे टुकड़ों में बांटें
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# 3. फ्री एम्बेडिंग मॉडल सेटअप
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. पाइनकोन डेटाबेस को सेट करें
PINECONE_INDEX_NAME = "medical-bot"

vectorstore = PineconeVectorStore.from_documents(
    docs, 
    embeddings, 
    index_name=PINECONE_INDEX_NAME,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

# 5. फ्री AI मॉडल लोड करें (Mistral AI)
repo_id = "Mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=os.environ.get("HF_TOKEN")
)

# मुख्य रूट - जो index.html पेज को लोड करेगा
@app.route("/")
def index():
    return render_template("index.html")

# चैट रूट - जो यूजर के सवाल का जवाब पाइनकोन और AI से लाकर देगा
@app.route("/get", methods=["GET"])
def chat():
    msg = request.args.get('msg')
    
    # पाइनकोन से मिलती-जुलती जानकारी ढूंढें
    docs = vectorstore.similarity_search(msg, k=1)
    
    if docs:
        context = docs[0].page_content
        # AI को निर्देश देना ताकि वह हिंदी में छोटा और सटीक जवाब दे
        prompt = f"तुम एक एक्सपर्ट मेडिकल असिस्टेंट हो। दी गई जानकारी के आधार पर मरीज को हिंदी में केवल 2 लाइन में छोटा और सीधा जवाब दो। जानकारी: {context}\n\nमरीज का सवाल: {msg}\nजवाब:"
        response = llm.invoke(prompt)
        return response
    else:
        return "मुझे इसके बारे में जानकारी नहीं मिली। कृपया डॉक्टर से संपर्क करें।"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)