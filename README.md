# IMDB-Conversational-Agent
This IMDB Chatbot is designed to provide natural language interaction with a structured movie dataset. It leverages a hybrid search strategy combining structured SQL and semantic retrieval via FAISS. A ReAct Agent implemented with LangGraph powers the chatbot, which enables dynamic decision-making, reasoning, conversational memory, and stepwise execution. The Streamlit front end provides an intuitive and interactive user experience with transparency regarding the ReAct Agent's thought process (Chain-of-Thought). The LLM of Choice powering the application is 'gemini-2.0-flash' ( will need Google API to replicate )
This report details the solution architecture, AI methodologies, data preparation techniques, and future improvements, explaining why this implementation is optimal for querying the IMDB dataset.

# How to Run
1. Clone the repo to local
2. Obtain a Gemini API Key. If you do not have you may get one from https://aistudio.google.com/apikey ( will need to log in with a google account )
3. Create a file at the root of the cloned repo and name it ".env"  
4. Inside the .env file write this (without the quotes):  "GOOGLE_API_KEY=<<API-KEY>>"  (replace <<API-KEY>> with the key obtained in step 2)
5. Install all dependencies for this project using the command: $ pip install -r requirements.txt
6. To run the application use the command: $ streamlit run app.py --server.fileWatcherType none

