# AI RAG Slack Agent for Organization-Specific Data Insights

An AI-powered Retrieval-Augmented Generation (RAG) Slack Agent that enables organizations to query and gain insights from their internal structured data through natural language conversations in Slack.

***

## Features

- Uses RAG architecture combining semantic vector search (FAISS + HuggingFace embeddings) with GPT-based reasoning  
- Classifies user queries to ensure relevant responses for data-related questions  
- Dynamically generates and executes Python pandas code to analyze structured data  
- Retrieves precise, context-aware answers from organization-specific datasets  
- Integrates seamlessly with Slack to provide real-time responses in threads  
- Keeps data and AI agent within organization’s environment for privacy and control  

***

## Technology Stack

- Python 3.9+  
- LangChain for RAG workflows  
- HuggingFace sentence-transformer embeddings  
- FAISS for fast vector similarity search  
- OpenAI GPT models (gpt-4o-mini) for query classification and response generation  
- Slack Bolt SDK for Slack API integration  
- pandas for data analysis and computation  

***

## Setup & Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/organization-rag-slack-agent.git
cd organization-rag-slack-agent
```

2. Setup Python environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables by creating a `.env` file:

```ini
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_APP_TOKEN=your-slack-app-token
OPENAI_API_KEY=your-openai-api-key
BOT_USER_ID=your-slack-bot-user-id
```

4. Prepare your internal structured data in a JSON format file (e.g., `data.json`) ready to be ingested by the bot.

5. Run the Slack bot:

```bash
python app.py
```

***

## Usage

- Invite and mention the bot in your Slack workspace.  
- Ask questions about your internal data in natural language.  
- The bot will fetch relevant data, analyze it, and provide clear, actionable insights as Slack messages.  

***

## Contributing

Contributions and improvements are welcome! Please open an issue or pull request to discuss your ideas.

***

## License

This project is licensed under the MIT License.

***
