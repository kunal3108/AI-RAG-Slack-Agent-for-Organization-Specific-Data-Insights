#!/usr/bin/env python
# coding: utf-8

################ Begin ##############

# Libraries
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import pandas as pd
import io
from contextlib import redirect_stdout

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Slack imports
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load environment variables
load_dotenv()

# Environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")      # xoxb- token
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")      # xapp- token
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        # OpenAI API key
BOT_USER_ID = os.getenv("BOT_USER_ID")              # Slack Bot User ID

# Check required env variables
if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Missing required environment variables.")

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)

# Load organization-specific data JSON file
file_path = "./data/data.json"

docs = []
with open(file_path, "r") as f:
    data = json.load(f)

for record in data:
    # Prepare summary or fallback to a basic description based on data fields
    summary = record.get("Summary") or json.dumps(record)
    docs.append(
        Document(
            page_content=summary.replace("?", ""),  # cleaned summary text
            metadata=record  # store full metadata for querying
        )
    )

# Create vector store for semantic search

# Split documents into chunks for better embedding context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Initialize sentence-transformer embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore from chunks
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Save FAISS index for reuse
vectorstore.save_local("./faiss_index")

print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks.")
print(f"FAISS index saved to './faiss_index/'")

# Global list to hold Slack event documents
slack_docs = []

# Allowed Slack channels (set your own channel IDs)
ALLOWED_CHANNELS = ["<YOUR-SLACK-CHANNEL-ID>"]  # Replace with your Slack channel IDs


# Function to embed slack message content
def embed_slack_message(doc):
    bot_mention = f"<@{BOT_USER_ID}>"
    cleaned_text = doc.page_content.replace(bot_mention, "").strip()
    embedding_vector = get_embedding(cleaned_text)
    print(f"Cleaned text for embedding: {cleaned_text}")
    return embedding_vector


def get_embedding(text):
    embedding_vector = embeddings.embed_query(text)
    print("Slack message embedded.")
    return embedding_vector


# Initialize OpenAI client
from openai import OpenAI
client = OpenAI()


# Function to classify user queries (disbursement/data query vs unknown)
def label_query_with_gpt(user_query: str) -> str:
    prompt = f"""
Classify the user question into one of two labels: "data query" or "unknown".

Example 1:
Q: How many orders were processed on July 1, 2025?
A: data query

Example 2:
Q: What was the total revenue in March 2025?
A: data query

Example 3:
Q: Who is the company CEO?
A: unknown

Question: {user_query}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
        stop=["\n"]
    )
    label = response.choices[0].message.content.strip().lower()
    print(f"Query classified as: {label}")
    return label


# Retrieve data schema from documents metadata
def get_current_schema(results):
    schema = {}
    for r in results:
        for field, value in r.metadata.items():
            value_type = type(value).__name__
            if field in schema:
                if schema[field] != value_type:
                    schema[field] = 'multiple'
            else:
                schema[field] = value_type
    return schema


# Compose pandas prompt for GPT
def make_pandas_prompt(user_question, schema, rows):
    schema_info = "\n".join(f"{k}: {v}" for k, v in schema.items())
    data_json = json.dumps(rows[:5], indent=2, default=str)
    prompt = f"""
You are a Python pandas expert working with structured data.
The data schema:
{schema_info}

Here are some example rows:
{data_json}

Write Python pandas code that, when run on a DataFrame called `df` built from the full dataset, answers the following user question:
\"\"\"{user_question}\"\"\"

Return only the necessary pandas code to compute the answer. After computing the value, please print the final result in a complete, natural language sentence suitable for presentation.

For example, print something like: "The total amount in July 2025 was 123456."
"""
    return prompt


# Safely execute generated pandas code
def safe_execute_pandas(pandas_code, rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    safe_locals = {'df': df}
    pandas_code = pandas_code.replace("```
    pandas_code = pandas_code.replace("```", "").strip()
    safe_globals = {'pd': pd}

    f = io.StringIO()
    with redirect_stdout(f):
        exec(pandas_code, safe_globals, safe_locals)
    output = f.getvalue().strip()
    return output


# Core query answering function
def run_query(user_query, docs, client):
    schema = get_current_schema(docs)
    rows = [doc.metadata for doc in docs]
    prompt = make_pandas_prompt(user_query, schema, rows)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    generated_code = response.choices[0].message.content
    answer = safe_execute_pandas(generated_code, rows)
    return answer


# Slack event listener
@app.event("message")
@app.event("app_mention")
@app.event("event_callback")
def handle_all_messages(event, say):
    print('Handling Slack event...')
    
    if event.get("subtype") == "bot_message":
        return

    channel_id = event.get("channel", "")
    if channel_id not in ALLOWED_CHANNELS:
        print(f"Not allowed channel: {channel_id}")
        return

    event_type = event.get("type", "")
    user_id = event.get("user", "")
    text = event.get("text", "")
    ts = event.get("ts")
    thread_ts = event.get("thread_ts", ts)

    print(f"\nEvent received at {time.strftime('%X')}:")
    print(f"type: {event_type}, channel: {channel_id}, user: {user_id}, ts: {ts}, thread_ts: {thread_ts}")
    print(f"text: {text}")

    bot_mention = f"<@{BOT_USER_ID}>"
    if bot_mention not in text:
        print("Bot not mentioned in the message, ignoring.")
        return

    # Avoid duplicate processing
    if any(doc.metadata.get("ts") == ts for doc in slack_docs):
        print(f"Duplicate event timestamp ({ts}), skipping.")
        return

    slack_doc = Document(
        page_content=text,
        metadata={
            "event_type": event_type,
            "channel_id": channel_id,
            "user_id": user_id,
            "ts": ts,
            "thread_ts": thread_ts,
            "full_event": event
        }
    )

    slack_docs.append(slack_doc)
    print(f"Document appended. Total Slack docs: {len(slack_docs)}")

    user_query = slack_doc.page_content
    label = label_query_with_gpt(user_query)

    unknown_reply = "Sorry, I do not have context to answer your query."

    if label == 'data query':
        answer = run_query(user_query, docs, client)
        say(text=answer, thread_ts=thread_ts)
    else:
        say(text=unknown_reply, thread_ts=thread_ts)

    print(f"Answer sent: {answer if label == 'data query' else unknown_reply}")

    
# Main entry point
if __name__ == "__main__":
    print("🤖 AI RAG Slack Agent is starting...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()

################ End ##############
