
# RAG based assistant as Slack App
# Santosh Gupta (santoshgupta@gmail.com)

# References:
# langchain: https://github.com/smaameri/multi-doc-chatbot
# slack: https://www.youtube.com/watch?v=Luujq0t0J7A

import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv(".env")

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith(".docx") or file.endswith(".doc"):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        text_path = "./docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
persist_directory = "./data"
model = "gpt-3.5-turbo"
# model = "gpt-4"
embedding = OpenAIEmbeddings()
if persist_directory and os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    vectordb = Chroma.from_documents(
        documents=documents, embedding=embedding, persist_directory=persist_directory
    )
    vectordb.persist()

# Detailed system instructions causing LLM response to exceed 8k token limits.
system_instruction = "Use provided context, otherwise do not make the answer up."

# Define your template with the system instruction
template = (
    f"{system_instruction} "
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)

# Create the prompt template
condense_question_prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0.7, model_name=model)

# create our Q&A chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    condense_question_prompt=condense_question_prompt,
    chain_type="stuff",
    return_source_documents=True,
    verbose=False,
)

chat_history = []
# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


# Message handler for Slack
@app.message(".*")
def message_handler(message, say, logger):
    # print(json.dumps(message, indent=4))
    user = message["user"]
    query = message["text"]
    handle_query(user, query, say)

@app.event("app_mention")
def handle_app_mention_events(event, say, logger):
    # print(json.dumps(event, indent=4))
    user = event["user"]
    query = event["text"]
    query = re.sub("\<@.*\>\s*", "", query)
    handle_query(user, query, say)

def handle_query(user, query, say):
    print(f"{user}: {query}")
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    output = result["answer"]
    chat_history.append((query, output))
    say(output)


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
