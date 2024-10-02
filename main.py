# Importing libs and modules
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel , Field
import re
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import shutil
from fastapi import UploadFile, File, HTTPException,Depends, status
from fastapi.responses import FileResponse

from twilio.twiml.messaging_response import MessagingResponse
from fastapi import FastAPI, Request, Response
import json
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)



from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from datetime import timedelta
from create_embeddings import create_vector_db 

# Setting Google API Key
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Path of vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'
# Path for vector database
DATA_PATH = 'data/'

@app.get("/")
async def read_root():
    return {"message": "Welcome to the St Patrick Hotel Chatbot API!"}

custom_prompt_template = """
    Use the following pieces of information to answer the user's question about Hotel St Patricks or St Patricks Hotel guest inquiries, requests and concerns.
    Your name is Guest chatbot, if you are greeted just greet and assist with required information. If the question is not about inquiries, requests and concerns at Hotel St. Patricks or is unclear, respond with: 'I can only answer questions related to St Patricks Hotel. How can I assist you?'
    Context: {context}
    Question: {question}
    
    Try to be conversational and give the best and correct answer.
    Be able to greet and resposnd to greetings and help requests conversationally.
    Helpful Answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Loading the model
def load_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6, max_output_tokens=2000)
    return llm

# Setting QA chain
def get_conversational_chain():
    prompt = set_custom_prompt()
    llm = load_llm()
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# Check if user input matches any of the categories
def user_input(user_question):

    # Default fallback to handle other cases via model (or DB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=False)
    
    return response

# Pydantic object
class Validation(BaseModel):
    prompt: str

# FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change this in production to restrict access
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API endpoint (POST Request)
@app.post("/llm_on_cpu")
async def final_result(item: Validation):
    response = user_input(item.prompt)
    return response



@app.post("/whatsapp_webhook")
async def whatsapp_webhook(request: Request):
    data = await request.form()
    user_message = data.get('Body')
    user_number = data.get('From')

    # Use your chatbot logic to generate a response
    response_message = user_input(user_message)

    # Send the response back to WhatsApp via Twilio
    twilio_response = MessagingResponse()
    twilio_response.message(response_message)

    return Response(str(twilio_response), media_type="application/xml")
