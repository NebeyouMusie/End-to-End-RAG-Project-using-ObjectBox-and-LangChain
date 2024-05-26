# End to End RAG Project using ObjectBox and LangChain
 - In this end to end project I have built a RAG app using ObjectBox Vector Databse and LangChain. RAG techniques allow us to augment a language model's knowledge base actively, ensuring your AI can access and reason with your data and the very latest information. With ObjectBox you can do that, without the data ever needing to leave the device.

![Streamlit Web App Interface](./images/RAG%20app%20UI.png)

## DEMO
 - You can check the project live [here](https://8512-01hwj8ynshjz7spkr595x77ec2.cloudspaces.litng.ai/)

## Description
- This project showcase the implementation of an advanced RAG system that uses Objectbox vectordatabse and Groq's LLAM3 model as an llm to retrieve information from different PDF documents.

Steps I followed:
1. I have used the `PyPdfDirectoryLoader` from the `langchain_community` document loader to load the PDF documents from the `us-census-data` directory.
2. transformed each text into a chunk of `1000` using the `RecursiveCharacterTextSplitter` imported from the `langchain.text_splitter`
3. stored the vector embeddings which were made using the `HuggingFaceBgeEmbeddings` using the `Objectbox` vector store.
4. setup the llm `ChatGroq` with the model name `Llama3-8b-8192`
5. Setup `ChatPromptTemplate`
6. Setup `vector_embedding` function to enbedd the documents and store them in the `Objectbox` vectorstore
7. finally created the `document_chain` and `retrieval_chain` for chaining llm to prompt and `retriever` to `document_chain` respectively

## Libraries Used

 - langchain==0.1.20
 - langchain-community==0.0.38
 - langchain-core==0.1.52
 - langchain-groq==0.1.3
 - langchain-objectbox
 - python-dotenv==1.0.1
 - pypdf==4.2.0

## Installation
 1. Prerequisites
    - Git
    - Command line familiarity
 2. Clone the Repository: `git clone https://github.com/NebeyouMusie/End-to-End-RAG-Project-using-ObjectBox-and-LangChain.git`
 3. Create and Activate Virtual Environment (Recommended)
    - `python -m venv venv`
    - `source venv/bin/activate`
 4. Navigate to the projects directory `cd ./End-to-End-RAG-Project-using-ObjectBox-and-LangChain` using your terminal
 5. Install Libraries: `pip install -r requirements.txt`
 6. Navigate to the app directory `cd ./app` using your terminal 
 7. run `streamlit run app.py`
 8. open the link displayed in the terminal on your preferred browser
 9. click on the `Embedd Documents` button and wait until the documnets are processed
 10. Enter your question from the PDFs found in the `us-census-data` directory

## Collaboration
- Collaborations are welcomed ❤️

## Acknowledgments
 - I would like to thank [Krish Naik](https://www.youtube.com/@krishnaik06)
   
## Contact
 - LinkedIn: [Nebeyou Musie](https://www.linkedin.com/in/nebeyou-musie)
 - Gmail: nebeyoumusie@gmail.com
 - Telegram: [Nebeyou Musie](https://t.me/NebeyouMusie)


