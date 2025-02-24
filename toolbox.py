from langchain_core.tools import tool
import prompts as p
import json
from datastore_setup import GetDataStoreAssets
import sqlite3
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import constants as c
from dotenv import load_dotenv

# Load data store objects
data_objs = GetDataStoreAssets().get_data_objects()
# Load environment variables from .env file
load_dotenv()

def clean_sql_query(query: str) -> str:
    if query is not None and len(query) > 0:
        clean_query = query.replace("`", "").replace("sql", "").strip()
        return clean_query
    raise ValueError("Malformed SQL Query")

def generate_sql_query(user_input: str) -> str:
    """
    Generates a valid SQL query using LangChain's structured prompt templates.
    Ensures memory is injected for context resolution.
    """
    # LLM Chain for SQL generation (TODO: move this out of this function)
    llm = ChatGoogleGenerativeAI(
        model=c.LLM_MODEL,  # model of your choice from gemini
        temperature=0,
        timeout=180,
        max_retries=2,
    )
    sql_chain = p.sql_query_gen_sys_prompt | llm
    # Retrieve last n messages from conversation memory
    # chat_history = chat_memory.load_memory_variables({}).get("history", [])
    # last_n_messages = context_len*(-1)
    # past_context = "\n".\
    #     join([msg["content"] for msg in chat_history][-10:]) if chat_history else "No prior context."

    past_context = "No prior context."
    contextual_query = f"Conversation Context: {past_context}\nUser Query: {user_input}"
    # Generate SQL using LangChain LLMChain
    response = sql_chain.invoke(input={"conversation_history": past_context, "user_query": user_input},
                                 config={"thread_id": "1"})
    sql_query = response.content
    return sql_query.strip()


@tool("structured-query-tool")
def adaptive_structured_query_tool(user_input: str) -> str:
    """
    Calls the LLM to generate a valid SQL query, then executes it in SQLite.
    This tool allows for selecting, filtering, sorting, grouping, ranking and partitioning the data in the SQLite database.
    Uses error handling to avoid common SQL execution issues.
    The Following are the Columns that are available in the SQL Datastore:
                  - `Poster_Link` (TEXT) -> An hyperlink to an image of the movie's poster
                  - `Series_Title` (TEXT) → Movie name
                  - `Released_Year` (INTEGER) → Year of release
                  - `Certificate` (TEXT) → Age rating
                  - `Runtime` (INTEGER) → Duration of the movie in minutes (e.g., '120' means 120 minutes)
                  - `Genre` (TEXT) → Movie genre(s)
                  - `IMDB_Rating` (FLOAT) → IMDb rating
                  - `Overview` (TEXT) → Short movie summary or description of the movie, basically it's plot
                  - `Meta_score` (FLOAT) → Metacritic score
                  - `Director` (TEXT) → Director’s name
                  - `Star1`, `Star2`, `Star3`, `Star4` (TEXT) → Lead actors (the movie's lead cast)
                  - `No_of_votes` (INTEGER) → Total votes
                  - `Gross` (FLOAT) → Box office earnings in US dollars
    """
    try:
        sql_query = generate_sql_query(user_input)  # Generate SQL from LLM
        sql_query = clean_sql_query(sql_query)  # Clean up potential markdown artifacts

        print(f" Executing SQL: {sql_query}")  # Debugging step

        conn = sqlite3.connect(data_objs["sqlite_db_path"])
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        results = [dict(zip(col_names, row)) for row in rows]

        if len(results) == 0:
            return json.dumps({"message": "No results found."})

        return json.dumps(results)

    except sqlite3.OperationalError as e:
        return json.dumps({"error": str(e), "query": sql_query})  # Return error details



def expand_semantic_query(user_query: str) -> str:
    """Expands the user query using predefined synonyms for better FAISS retrieval."""
    SYNONYMS = {
        "death": ["dying", "murder", "dead", "kill", "fatal"],
        "dream": ["dreams", "subconscious", "sleep"],
        "robot": ["android", "AI", "machine", "cyborg"]
    }

    lower_q = user_query.lower()
    expansions = []

    for keyword, syns in SYNONYMS.items():
        if keyword in lower_q:
            expansions.extend(syns)

    if expansions:
        expansion_str = " OR ".join(expansions)
        return f"{user_query} ({expansion_str})"

    return user_query


@tool("semantic-search-tool")
def adaptive_semantic_search_tool(search_query: str) -> str:
    """
    Looks up movie overviews and plots using Semantic Search on FAISS Vectorstore based on the search_query.
    The search_query should be formulated efficiently from the original user_query for semantic search.
    This tool uses SQLite datastore to lookup movie titles.
    FAISS search with adaptive expansion if empty, using SQLite movie titles.
    The Following are the Movie Data Columns that were merged to create each embedding in the FAISS index:
                  - `Overview` (TEXT) → Short movie summary (Plot or Overview or Description)
    """

    # Load embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Retrieve all movie titles from the SQLite database
    conn = sqlite3.connect(data_objs["sqlite_db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT distinct Series_Title FROM movies")
    movie_data = cursor.fetchall()  # Returns list of tuples (id, title)

    if not movie_data:
        return json.dumps([])  # No data available

    # # Retrieve memory
    # last_n_messages = context_len * (-1)
    # chat_history = chat_memory.load_memory_variables({})["history"]
    # past_context = "\n".join([msg["content"] for msg in chat_history][last_n_messages:])

    past_context = ""

    # Modify user query with past memory
    # contextual_query = f"Conversation Context: {past_context}\nUser Query: {search_query}"

    # Extract movie titles & create mapping
    movie_titles = [row[0] for row in movie_data]  # Extract titles
    # movie_ids = [row[0] for row in movie_data]  # Extract corresponding IDs
    # id_to_title_map = {idx: title for idx, title in zip(range(len(movie_titles)), movie_titles)}
    id_to_title_map = {}
    idx = 1
    for title in movie_titles:
        id_to_title_map[idx] = title
        idx+=1

    # Perform FAISS search
    q_emb = embedding_model.encode([search_query])
    distances, indices = data_objs["faiss_index"].search(q_emb, 5)  # Get top 5 matches
    matched_titles = [id_to_title_map[idx] for idx in indices[0] if idx in id_to_title_map]

    # If no matches, try expanding the query with synonyms
    if not matched_titles:
        expanded_query = expand_semantic_query(search_query)
        q_emb_exp = embedding_model.encode([expanded_query])
        distances, indices = data_objs["faiss_index"].search(q_emb_exp, 5)
        matched_titles = [id_to_title_map[idx] for idx in indices[0] if idx in id_to_title_map]

    return json.dumps(matched_titles)

# print(adaptive_semantic_search_tool("comedy with death and dead"))