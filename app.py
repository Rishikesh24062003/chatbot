import json
import google.generativeai as genai
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
import os
from ollama import chat
from ollama import ChatResponse
import gradio as gr
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu"
)


# Existing code for embeddings, model configurations, and utility functions
parent_retriever_k = 12
max_parents = 2
context_retriever_k = 3
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",model_kwargs={'trust_remote_code':True,'device' : 'cpu'})
# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running the application.")
generation_config = {
  "temperature": 0.1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 2048,
}

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash',generation_config=generation_config)

system_prompt = f"""
You are a help-full conversational Assistant chatbot from QuickSell that helps users to find information and response for their query from the documents.
Do not mention context or any other information in the response. Just provide the answer based on the context.

"""
chat_model = genai.GenerativeModel('gemini-2.0-flash',generation_config=generation_config,system_instruction=system_prompt)


def ollama_chat(query,history=[], streaming=False):
    system_prompt_ollama = {
    'role': 'system',
    'content': system_prompt
    }
    if history is None or len(history) == 0:
        history.append(system_prompt_ollama)
    if history[0] != system_prompt_ollama:
        history.insert(0,system_prompt_ollama)
    history.append({"role": "user", "content": query})
    if streaming:
        response: ChatResponse= chat(model='qwen2.5:7b', messages=history, stream=True)
        return response
    response: ChatResponse= chat(model='qwen2.5:7b', messages=history)
    return response['message']['content']


def llm(prompt):
    return model.generate_content(prompt).text

def get_json(text):
    try: 
        text = text.split("```json")[1]
        text = text.split("```")[0]
        return json.loads(text)
    except:
        return {"queries": None}


def read_pdf(file_path):
    if file_path.endswith('.pdf'):
        doc_loader = UnstructuredPDFLoader(file_path)
        docs = doc_loader.load()
    else: 
        raise ValueError("Invalid file format")
    return docs

def get_context(query, session_state):
    # retrieve parent chunk
    retrieved_context = []
    
    # Use the parent_retriever from session_state instead of a global variable
    parent_retriever = session_state.get("parent_retriever")
    parent_vector_store = session_state.get("vector_db")
    
    if parent_retriever is None or parent_vector_store is None:
        return [],retrieved_context
    
    parent_results = parent_retriever.invoke(query)
    if len(parent_results) == 0:
        return [],retrieved_context
    
    parent_ids = []
    for res in parent_results:
        parent_id = res.metadata['parent_id']
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    if len(parent_ids) > max_parents:
        parent_ids = parent_ids[:max_parents] 
    for parent_id in parent_ids:
        res_context = parent_vector_store.similarity_search(
            query,
            k=context_retriever_k,
            filter={"parent_id": parent_id, "isparent": False},
        )
        retrieved_context.extend(res_context)
    
    final_context = "    "
    for i,context in enumerate(retrieved_context):
        final_context += str(i)+": "+ context.page_content + "\n\n"
    return final_context,retrieved_context

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1800*5,
                                          chunk_overlap=100) # 5 pages

def split2parent(doc):
    chunks = parent_splitter.split_documents(doc)
    for i,chunk in enumerate(chunks):
        chunk.metadata['parent_id'] = i
        chunk.metadata['isparent'] = True
    return chunks

def split2context(docs):
    chunks = context_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata['parent_id'] = chunk.metadata['parent_id']
        chunk.metadata['isparent'] = False
    return chunks

def get_compressed_prompt(prompt):
  results = compressor.compress_prompt_llmlingua2(
    prompt,
    rate=0.4,
    force_reserve_digit=True,
    drop_consecutive=True)
  return results['compressed_prompt']


def get_prompt_for_parent_details(parent_chunk):
    prompt = f"""
    {parent_chunk.page_content}
    
    ### Instruction : For the given chunk of a document, provide concise one-line description for the each content that present in the chunk.
    Make sure to add entity names, dates, numbers, and other important information in the description.
    Give output in the following format:
    ```json
    {{
        "description": ["description content 1", "description content  2", "description content  3" ...]
    }}
    ```
    provide what is asked only in the above format.
    """
    return prompt



context_splitter = RecursiveCharacterTextSplitter(chunk_size=int(1800*0.5),
                                            chunk_overlap=50) # 0.5 pages

def split2context(docs):
    chunks = context_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata['isparent'] = False
        chunk.metadata['parent_id'] = chunk.metadata['parent_id']
    return chunks

def split_query(query,history=[],model="gemini"):
    prompt = f"""
    User chat history : {history[-6:]}
    user current query : {query}

    ### Your job is create new queries such that it can be used to get the correct context from the knowledge base using RAG that do
    semantic search and need little contextual query for better search.
    like : is there any Ashish ? then context can be "Person named "Ashish" is present in the document ?"
    ----------------
    Follow the given instructions : 
    if   : 
        the query is not clear or general conversation like "hi" or any other general conversation then ignore that query. and give output as "None" only.
    else : 
        1. Correct the user's query if there is any mistake and little context to the query.
        2. if multiple things ask in the user's current query and create new separate queries for each ask. 
        3. check if the users current query is related to the previous queries, if yes then use given users chat history to create new correct queries.
    
    Give output in the given format : 
    ```json
    {{"queries" : ["query -1", "query-2", ...]}}
    ```
    """
    if model == "gemini":
        res = llm(prompt)
    if model == "ollama":
        res = ollama_chat(prompt)
    return res


# Define the chat_memory class
class chat_memory:
    def __init__(self):
        self.memory = []
    
    def add_to_memory(self, query, response):
        self.memory.append({"user": query, "model": response})
    
    def get_memory(self):
        return self.memory

# Function to parse chat history for different models
def parse_history(history, model="gemini"):
    transformed_history = []
    if model == "gemini":
        for item in history:
            user = item['user']
            response = item['model']
            transformed_history.append({"role": "user", "parts": [{"text": user}]})
            transformed_history.append({"role": "model", "parts": [{"text": response}]})
    if model == "ollama":
        for item in history:
            user = item['user']
            response = item['model']
            transformed_history.append({"role": "user", "content": user})
            transformed_history.append({"role": "model", "content": response})
    return transformed_history

# Function to get prompt for parent details
def get_prompt_for_parent_details(parent_chunk):
    prompt = f"""
    {parent_chunk.page_content}
    
    ### Instruction : For the given chunk of a document, provide concise one-line description for the each content that present in the chunk.
    Make sure to add entity names, dates, numbers, and other important information in the description.
    Give output in the following format:
    ```json
    {{
        "description": ["description content 1", "description content  2", "description content  3" ...]
    }}
    ```

    Do not create multiple queries asking for the same thing .
    provide what is asked only in the above format.
    """
    return prompt

# Function to get the final prompt for the chatbot
def get_prompt(query, history=[], model="gemini", session_state=None):
    queries = split_query(query, history, model)
    queries = get_json(queries)['queries']
    
    if queries is None or len(queries) == 0 or queries == ['None']:
        prompt = f"user query : {query} \n\n you can ask user to clarify the query if needed."
        return prompt,[]
    contexts = []
    sources =[queries]
    for temp_query in queries:
        context,source_ = get_context(temp_query,session_state=session_state)
        compressed_ = get_compressed_prompt(context)
        contexts.append(compressed_)
        sources.extend(source_)
    
    contexts = list(set(contexts))
    prompt = f"""
    user's query or response : {query}
    ----------------
    context for each part of query : {contexts}
    
    Just provide the answer based on the context. Reply as a Assistant chatbot from QuickSell.
    ### Important : And if the context is not present or not relevant do not explain just give output as "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
    .
    """
    return prompt,sources

# Function to handle PDF upload and vector database creation
def upload_pdf(pdf_file, session_state, chatbot, query_textbox, source_documents_textbox, show_sources_checkbox, database_option):
    if pdf_file is None:
        return session_state, chatbot, query_textbox, source_documents_textbox
    
    file_path = pdf_file.name
    

    if database_option == "Load Existing":
        try:
            parent_vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
            parent_retriever = parent_vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": parent_retriever_k, "filter": {"isparent": True}}
            )
        except Exception as e:

            gr.Warning(f"Failed to load existing database. Creating a new one. Error: {str(e)}")
            database_option = "Create New"
    

    if database_option == "Create New":
        docs = read_pdf(file_path)
        parent_chunks = split2parent(docs)
        parent_document = []
        for parent_chunk in parent_chunks:
            parent_details = llm(get_prompt_for_parent_details(parent_chunk))
            parent_details = get_json(parent_details)['description']
            for description in parent_details:
                parent_document.append(
                    Document(
                        page_content=description,
                        metadata=parent_chunk.metadata
                    )
                )
        
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        parent_vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        uuids = [str(uuid4()) for _ in range(len(parent_document))]
        parent_vector_store.add_documents(documents=parent_document, ids=uuids)
        
        context_chunks = split2context(parent_chunks)
        parent_vector_store.add_documents(documents=context_chunks, ids=[str(uuid4()) for _ in range(len(context_chunks))])
    
    parent_vector_store.save_local("faiss_index")
    
    # Create retriever
    parent_retriever = parent_vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": parent_retriever_k, "filter": {"isparent": True}}
    )
    

    session_state["vector_db"] = parent_vector_store
    session_state["parent_retriever"] = parent_retriever
    session_state["chat_memory"] = chat_memory()
    session_state["model"] = session_state.get("model", "gemini")
    session_state["pdf_processed"] = True
    
    chatbot = []
    source_documents_textbox = ""
    
    return session_state, chatbot, query_textbox, source_documents_textbox

# Function to generate chatbot response
def generate_response(query, session_state, chatbot, show_sources_checkbox):
    if session_state.get("pdf_processed", False):
        model = session_state.get("model", "gemini")
        chat_mem = session_state.get("chat_memory")
        prompt,source_docs = get_prompt(query, chat_mem.get_memory(), model,session_state=session_state)
        if model == "gemini":
            chatbot_instance = chat_model.start_chat(history=parse_history(chat_mem.get_memory(), model="gemini"))
            response = chatbot_instance.send_message(prompt).text
        elif model == "ollama":
            response = ollama_chat(prompt, history=parse_history(chat_mem.get_memory(), model="ollama"))
        chat_mem.add_to_memory(query=query, response=response)
        
        session_state["latest_source_docs"] = source_docs
        chatbot.append((query, response))
        #chatbot.append(("bot", response))
        if show_sources_checkbox:
            source_documents_text = str(source_docs)
        else:
            source_documents_text = ""
        return chatbot, source_documents_text
    else:
        chatbot.append(("user", query))
        chatbot.append(("bot", "Please upload a PDF first."))
        return chatbot, ""


def render_source_documents(session_state, show_sources_checkbox):
    if show_sources_checkbox:
        source_docs = session_state.get("latest_source_docs", [])
        source_documents_text = str(source_docs)
    else:
        source_documents_text = ""
    return source_documents_text


def reset_session(session_state, chatbot, source_documents_textbox):
    session_state.clear()
    chatbot = []
    source_documents_textbox = ""
    return session_state, chatbot, source_documents_textbox

def update_model(model, session_state):
    session_state["model"] = model


def main():
    with gr.Blocks() as demo:
        session_state = gr.State({})
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(choices=["gemini", "ollama"], label="Select Model", value="gemini")
                database_dropdown = gr.Dropdown(
                    choices=["Create New", "Load Existing"], 
                    label="Database Option", 
                    value="Create New"
                )
                pdf_uploader = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                

                show_sources_checkbox = gr.Checkbox(label="Show source documents", value=False)
                source_documents_textbox = gr.Textbox(label="Source Documents", interactive=False)
                reset_button = gr.Button("Reset Session")
            
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="QuickSell Document Chat",
                    height=600,
                    render=True
                )

                with gr.Row():
                    query_textbox = gr.Textbox(
                        label="Enter your query", 
                        placeholder="Type your question about the uploaded document...",
                        container=False,
                        scale=8
                    )
                    submit_button = gr.Button("Submit", variant="primary",size="lg",scale=2)
        

        pdf_uploader.upload(
            fn=upload_pdf,
            inputs=[pdf_uploader, session_state, chatbot, query_textbox, source_documents_textbox, show_sources_checkbox, database_dropdown],
            outputs=[session_state, chatbot, query_textbox, source_documents_textbox]
        )

        submit_button.click(
            fn=generate_response,
            inputs=[query_textbox, session_state, chatbot, show_sources_checkbox],
            outputs=[chatbot, source_documents_textbox],
            api_name="submit_query"
        ).then(
            fn=lambda x: gr.update(value=""),  
            inputs=query_textbox,
            outputs=query_textbox
        )
        

        model_dropdown.change(
            fn=update_model,
            inputs=[model_dropdown, session_state],
            outputs=None
        )
        

        reset_button.click(
            fn=reset_session,
            inputs=[session_state, chatbot, source_documents_textbox],
            outputs=[session_state, chatbot, source_documents_textbox]
        )
        

        show_sources_checkbox.change(
            fn=render_source_documents,
            inputs=[session_state, show_sources_checkbox],
            outputs=source_documents_textbox
        )
    
    demo.launch(
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == '__main__':
    main()
