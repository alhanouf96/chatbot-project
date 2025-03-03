import streamlit as st
import uuid
import requests

# Backend URLs define
LOAD_CHAT_URL = "http://127.0.0.1:5000/load_chat/"
SAVE_CHAT_URL = "http://127.0.0.1:5000/save_chat/"
DELETE_CHAT_URL = "http://127.0.0.1:5000/delete_chat/"
UPLOAD_PDF_URL = "http://127.0.0.1:5000/upload_pdf/"
CHAT_URL = "http://127.0.0.1:5000/chat/"
RAG_CHAT_URL = "http://127.0.0.1:5000/rag_chat/"

# Initialize session state
if "history_chats" not in st.session_state:
    st.session_state["history_chats"] = []
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None
if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = {}

# Functions to manage chats
def load_chats_from_db():
    response = requests.get(LOAD_CHAT_URL, timeout=10)

    if response.status_code == 200:
        records = response.json()
        for record in records:
            chat_id = record['id']
            messages = record['messages']
            name = record['chat_name']
            pdf_path = record['pdf_path']
            pdf_name = record['pdf_name']
            pdf_uuid = record['pdf_uuid']
            st.session_state["history_chats"].append({"id": chat_id, "messages": messages, "pdf_name":pdf_name, "pdf_path":pdf_path, "pdf_uuid":pdf_uuid})
            st.session_state["chat_names"][chat_id] = name
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

def save_chat_to_db(chat_id, chat_name, messages, pdf_name, pdf_path, pdf_uuid):
    payload = {
                "chat_id": chat_id,
                "chat_name": chat_name,
                "messages": messages,
                "pdf_name": pdf_name,
                "pdf_path": pdf_path,
                "pdf_uuid": pdf_uuid
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(SAVE_CHAT_URL, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Failed to save data. Status code: {response.status_code}")

def create_chat_with_pdf(chat_name, uploaded_pdf):

    with st.spinner("Uploading and Processing document, please wait..."):
        files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}

        response = requests.post(UPLOAD_PDF_URL, files=files)

        if response.status_code == 200:

            pdf_path = response.json()["pdf_path"]
            pdf_uuid = response.json()["pdf_uuid"]

            new_chat_id = str(uuid.uuid4())
            new_chat = {"id": new_chat_id, "messages": [], "pdf_name":uploaded_pdf.name, "pdf_path": pdf_path, "pdf_uuid":pdf_uuid}
            st.session_state["history_chats"].insert(0, new_chat)
            st.session_state["chat_names"][new_chat_id] = chat_name
            st.session_state["current_chat"] = new_chat_id
            save_chat_to_db(new_chat_id, chat_name, [], uploaded_pdf.name, pdf_path, pdf_uuid)
            st.success("Successed!")
        else:
            st.error("Failed to upload PDF.")

def create_chat(chat_name):
    new_chat_id = str(uuid.uuid4())
    new_chat = {"id": new_chat_id, "messages": [], "pdf_name":None, "pdf_path": None, "pdf_uuid": None}
    st.session_state["history_chats"].insert(0, new_chat)
    st.session_state["chat_names"][new_chat_id] = chat_name
    st.session_state["current_chat"] = new_chat_id
    
    save_chat_to_db(new_chat_id, chat_name, [], None, None, None)
    

def delete_chat():
    if st.session_state["current_chat"]:
        chat_id = st.session_state["current_chat"]
        st.session_state["history_chats"] = [
            chat for chat in st.session_state["history_chats"] if chat["id"] != chat_id
        ]
        del st.session_state["chat_names"][chat_id]
        payload = {
                "chat_id": chat_id
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(DELETE_CHAT_URL, json=payload, headers=headers)

        if response.status_code != 200:
            print(f"Failed to delete data. Status code: {response.status_code}")

        st.session_state["current_chat"] = (
            st.session_state["history_chats"][0]["id"] if st.session_state["history_chats"] else None
        )

def select_chat(chat_id):
    st.session_state["current_chat"] = chat_id

# Load chats from database
load_chats_from_db()
# st.write(st.session_state)
# Sidebar
with st.sidebar:
    st.title("Chat Management")

    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    chat_name = st.text_input("Enter Chat Name:", key="new_chat_name")

    if st.button("Create New Chat"):
        if chat_name.strip():
            create_chat(chat_name.strip())
        else:
            st.warning("Chat name cannot be empty.")

    if st.button("Create New Chat with PDF"):
        if not uploaded_pdf:
            st.warning("Please upload a PDF file before creating the chat.")
        elif chat_name.strip():
            create_chat_with_pdf(chat_name.strip(), uploaded_pdf)
        else:
            st.warning("Chat name cannot be empty.")

    # if not st.session_state["current_chat"]:
    #     st.session_state["current_chat"] = st.session_state["history_chats"][0]['id']

    if st.session_state["history_chats"]:
        chat_options = {
            chat["id"]: st.session_state["chat_names"][chat["id"]]
            for chat in st.session_state["history_chats"]
        }
        selected_chat = st.radio(
            "Select Chat",
            options=list(chat_options.keys()),
            format_func=lambda x: chat_options[x],
            # index=list(chat_options.keys()).index(st.session_state["current_chat"]),
            key="chat_selector",
            on_change=lambda: select_chat(st.session_state.chat_selector),
        )
        st.session_state["current_chat"] = selected_chat

        st.button("Delete Chat", on_click=delete_chat)

# Main Content
st.title("Chatbot Application")

if st.session_state["current_chat"]:
    chat_id = st.session_state["current_chat"]
    chat_name = st.session_state["chat_names"][chat_id]
    st.subheader(f"Current Chat: {chat_name}")

    current_chat = next(
        (chat for chat in st.session_state["history_chats"] if chat["id"] == chat_id),
        None,
    )

    if current_chat:
        if current_chat["pdf_name"]:
            pdf_name = current_chat["pdf_name"]
            st.subheader(f"Associate with: {pdf_name}")

        for message in current_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Your Message:"):
            current_chat["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                payload = {
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in current_chat["messages"]
                    ]
                }
                headers = {"Content-Type": "application/json"}

                if current_chat["pdf_uuid"]:
                    payload["pdf_uuid"] = current_chat["pdf_uuid"]
                    chat_taret_url = RAG_CHAT_URL
                else:
                    chat_taret_url = CHAT_URL

                # No Stream approach
                # stream = requests.post(chat_url, json=payload, headers=headers)
                # response = stream.json()["reply"]
                # st.markdown(response)

                # Stream approach
                def get_stream_response():
                    with requests.post(chat_taret_url, json=payload, headers=headers, stream=True) as r:
                        for chunk in r:
                            yield chunk.decode("utf-8")

                response = st.write_stream(get_stream_response)
                current_chat["messages"].append({"role": "assistant", "content": response})
                save_chat_to_db(chat_id, chat_name, current_chat["messages"], current_chat["pdf_name"], current_chat["pdf_path"], current_chat["pdf_uuid"])
else:
    st.write("No chat selected. Use the sidebar to create or select a chat.")
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import json
import psycopg2
import os
import uuid
from psycopg2.extras import RealDictCursor
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
import chromadb

load_dotenv()

DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

model = "gpt-3.5-turbo"

# VECTOR_DB_DIR = "chromadb"
# os.makedirs(VECTOR_DB_DIR, exist_ok=True)

llm = ChatOpenAI(model=model)

# LangChain setup
embedding_function = OpenAIEmbeddings()
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_or_create_collection("langchain")
vectorstore = Chroma(
    client=chroma_client,
    collection_name="langchain",
    embedding_function=embedding_function,
)


app = FastAPI()

# Request models
class ChatRequest(BaseModel):
    messages: List[dict]

class SaveChatRequest(BaseModel):
    chat_id: str
    chat_name: str
    messages: List[dict]
    pdf_name: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_uuid: Optional[str] = None

class DeleteChatRequest(BaseModel):
    chat_id: str

class RAGChatRequest(BaseModel):
    messages: List[dict]
    pdf_uuid: str

# Dependency to manage database connection
def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=request.messages,
            stream=True,
        )

        # if you don't want to stream the output
        # set the stream parameter to False in above function
        # and uncommnet the belowing line
        # return {"reply": response.choices[0].message.content}

        # Function to send out the stream data
        def stream_response():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        # Use StreamingResponse to return
        return StreamingResponse(stream_response(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load_chat/")
async def load_chat(db: psycopg2.extensions.connection = Depends(get_db)):
    try:
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT id, name, file_path, pdf_name, pdf_path, pdf_uuid FROM advanced_chats ORDER BY last_update DESC")
            rows = cursor.fetchall()

        records = []
        for row in rows:
            chat_id, name, file_path, pdf_name, pdf_path, pdf_uuid= row["id"], row["name"], row["file_path"], row["pdf_name"], row["pdf_path"], row["pdf_uuid"]
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                records.append({"id": chat_id, "chat_name": name, "messages": messages, "pdf_name":pdf_name, "pdf_path":pdf_path, "pdf_uuid":pdf_uuid})

        return records

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/save_chat/")
async def save_chat(request: SaveChatRequest, db: psycopg2.extensions.connection = Depends(get_db)):
    try:
        file_path = f"chat_logs/{request.chat_id}.json"
        os.makedirs("chat_logs", exist_ok=True)
        
        # Save messages to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(request.messages, f, ensure_ascii=False, indent=4)
        
        # Insert or update database record
        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO advanced_chats (id, name, file_path, last_update, pdf_path, pdf_name, pdf_uuid)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET name = EXCLUDED.name, file_path = EXCLUDED.file_path, last_update = CURRENT_TIMESTAMP, pdf_path = EXCLUDED.pdf_path, pdf_name = EXCLUDED.pdf_name, pdf_uuid = EXCLUDED.pdf_uuid
                """,
                (request.chat_id, request.chat_name, file_path, request.pdf_path, request.pdf_name, request.pdf_uuid),
            )
        db.commit()
        return {"message": "Chat saved successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/delete_chat/")
async def delete_chat(request: DeleteChatRequest, db: psycopg2.extensions.connection = Depends(get_db)):
    try:
        # Retrieve the file path before deleting the record
        file_path = None
        with db.cursor() as cursor:
            cursor.execute("SELECT file_path FROM advanced_chats WHERE id = %s", (request.chat_id,))
            result = cursor.fetchone()
            if result:
                file_path = result[0]
            else:
                raise HTTPException(status_code=404, detail="Chat not found")

        # Delete the record from the database
        with db.cursor() as cursor:
            cursor.execute("DELETE FROM advanced_chats WHERE id = %s", (request.chat_id,))
        db.commit()

        # Delete the associated file, if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return {"message": "Chat deleted successfully"}

    except HTTPException:
        # Reraise known exceptions
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        pdf_uuid = str(uuid.uuid4())
        file_path = f"pdf_store/{pdf_uuid}_{file.filename}"
        os.makedirs("pdf_store", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and process PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Add to ChromaDB
        vectorstore.add_texts(
            [doc.page_content for doc in texts], 
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=[{"pdf_uuid": pdf_uuid} for _ in texts]    
        )

        return {"message": "File uploaded successfully", "pdf_path": file_path, "pdf_uuid":pdf_uuid}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/rag_chat/")
async def rag_chat(request: RAGChatRequest):

    retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"pdf_uuid": request.pdf_uuid}}
        )
    
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []

    user_input = request.messages[-1]
    previous_chat = request.messages[:-1]

    for message in request.messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        if message["role"] == "assistant":
            chat_history.append(AIMessage(content=message["content"]))
    
    # response = rag_chain.invoke({
    #     "chat_history":chat_history,
    #     "input":user_input
    # })

    chain = rag_chain.pick("answer")

    stream = chain.stream({
        "chat_history":chat_history,
        "input":user_input
    })

    def stream_response():
            for chunk in stream:
                yield chunk

    # Use StreamingResponse to return
    return StreamingResponse(stream_response(), media_type="text/plain")
