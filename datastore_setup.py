import os
import sqlite3
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union


def get_structured_dataset(df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and structures a DataFrame by handling missing values and setting appropriate data types.

    Args:
        df_ret (pd.DataFrame): The DataFrame containing movie data.

    Returns:
        pd.DataFrame: The cleaned and structured DataFrame.
    """
    try:
        # Define numeric column types
        float_cols = ["IMDB_Rating", "Meta_score", "Gross"]
        integer_cols = ["Runtime", "No_of_Votes", "Released_Year"]

        # Fill missing values for non-numeric columns
        for _col in df_ret.columns:
            if _col not in float_cols and _col not in integer_cols:
                df_ret[_col] = df_ret[_col].fillna("Not Available")

        # Cleanup numeric columns
        if "Runtime" in df_ret.columns:
            df_ret["Runtime"] = df_ret["Runtime"].str.extract(r"(\d+)")  # Extract only digits
        if "Gross" in df_ret.columns:
            df_ret["Gross"] = df_ret["Gross"].str.replace(",", "", regex=True)  # Remove commas

        # Convert to appropriate types
        for _col in float_cols + integer_cols:
            df_ret[_col] = pd.to_numeric(df_ret[_col], errors="coerce").fillna(-999)
            df_ret[_col] = df_ret[_col].astype(float if _col in float_cols else int)

        return df_ret

    except Exception as e:
        raise ValueError(f"Error structuring dataset: {e}")


def setup_databases(
        csv_search_paths: List[str],
        sqlite_search_paths: List[str],
        faiss_search_paths: List[str],
        sqlite_db_name: str = "movies.db",
        faiss_index_name: str = "movies.index",
        csv_filename: str = "IMDB_data.csv"
) -> Dict[str, Union[str, faiss.IndexFlatL2]]:
    """
    Sets up SQLite and FAISS databases by either loading existing ones or creating new ones from CSV data.

    Args:
        csv_search_paths (List[str]): Paths to search for CSV files.
        sqlite_search_paths (List[str]): Paths to search for SQLite databases.
        faiss_search_paths (List[str]): Paths to search for FAISS indexes.
        sqlite_db_name (str): SQLite database filename.
        faiss_index_name (str): FAISS index filename.
        csv_filename (str): CSV filename containing movie data.

    Returns:
        Dict[str, Union[str, faiss.IndexFlatL2]]: Paths and FAISS index if successful.
    """

    try:
        # Step A. Check for existing SQLite DB
        existing_sqlite_path = next(
            (os.path.join(path_dir, sqlite_db_name) for path_dir in sqlite_search_paths if os.path.isfile(os.path.join(path_dir, sqlite_db_name))),
            None
        )

        # Step B. Check for existing FAISS index
        existing_faiss_path = next(
            (os.path.join(path_dir, faiss_index_name) for path_dir in faiss_search_paths if os.path.isfile(os.path.join(path_dir, faiss_index_name))),
            None
        )

        # Step C. If both exist, load them
        if existing_sqlite_path and existing_faiss_path:
            print(f"[INFO] Found existing SQLite DB at: {existing_sqlite_path}")
            print(f"[INFO] Found existing FAISS index at: {existing_faiss_path}")
            index = faiss.read_index(existing_faiss_path)
            return {"sqlite_db_path": existing_sqlite_path, "faiss_index": index}

        # Step D. Locate CSV if databases are missing
        existing_csv_path = next(
            (os.path.join(path_dir, csv_filename) for path_dir in csv_search_paths if os.path.isfile(os.path.join(path_dir, csv_filename))),
            None
        )

        if not existing_csv_path:
            raise FileNotFoundError(
                f"Could not find CSV in provided csv_search_paths: {csv_search_paths} "
                f"nor found existing DB/FAISS. Aborting!"
            )

        print(f"[INFO] CSV found at: {existing_csv_path}")
        df = pd.read_csv(existing_csv_path)

        # Step E. Create SQLite DB if missing
        if not existing_sqlite_path:
            if not sqlite_search_paths:
                raise ValueError("No sqlite_search_paths provided. Cannot create a new SQLite DB.")
            db_output_dir = sqlite_search_paths[0]
            os.makedirs(db_output_dir, exist_ok=True)
            new_db_path = os.path.join(db_output_dir, sqlite_db_name)
            print(f"[INFO] Creating new SQLite DB at {new_db_path}")
            df_structured = get_structured_dataset(df)
            conn = sqlite3.connect(new_db_path)
            df_structured.to_sql("movies", conn, if_exists="replace", index=False)
        else:
            conn = sqlite3.connect(existing_sqlite_path)
            new_db_path = existing_sqlite_path

        # Step F. Create FAISS index if missing
        if not existing_faiss_path:
            if not faiss_search_paths:
                raise ValueError("No faiss_search_paths provided. Cannot create a new FAISS index.")
            faiss_output_dir = faiss_search_paths[0]
            os.makedirs(faiss_output_dir, exist_ok=True)
            new_faiss_path = os.path.join(faiss_output_dir, faiss_index_name)
            print(f"[INFO] Creating new FAISS index at {new_faiss_path}")

            # Columns for semantic search
            columns_for_indexing = ['Series_Title', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Overview']
            missing_cols = [col for col in columns_for_indexing if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV is missing required columns for FAISS index: {missing_cols}")

            # Fill missing values
            df[columns_for_indexing] = df[columns_for_indexing].fillna(" ")
            df["semantic_search_data"] = df["Overview"]

            # Create embeddings
            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vectors = np.array(embed_model.encode(df["semantic_search_data"].tolist(), show_progress_bar=True), dtype=np.float32)

            # Build FAISS index
            dim = vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            faiss.write_index(index, new_faiss_path)
        else:
            new_faiss_path = existing_faiss_path
            index = faiss.read_index(new_faiss_path)

        print("[INFO] Setup complete.")
        return {"sqlite_db_path": new_db_path, "faiss_index": index}

    except Exception as e:
        raise RuntimeError(f"Database setup error: {e}")


class GetDataStoreAssets:
    """
    Initializes the data store assets by setting up the SQLite and FAISS databases.
    """

    def __init__(self):
        self.data_asset_paths = None

        csv_paths = ["./data", "/data/csv_files"]
        sqlite_paths = ["./db", "/databases/sqlite"]
        faiss_paths = ["./db", "/faiss_indexes"]

        try:
            self.data_asset_paths = setup_databases(
                csv_search_paths=csv_paths,
                sqlite_search_paths=sqlite_paths,
                faiss_search_paths=faiss_paths,
                sqlite_db_name="movies.db",
                faiss_index_name="movies.index",
                csv_filename="IMDB_data.csv"
            )
            print("!!! Structured & Unstructured Databases Connected !!!")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise ConnectionError("Data Setup Not Complete :: " + str(e))

    def get_data_objects(self) -> Dict[str, Union[str, faiss.IndexFlatL2]]:
        """
        Returns the paths to the databases.

        Returns:
            Dict[str, Union[str, faiss.IndexFlatL2]]: Paths to SQLite and FAISS index.
        """
        return self.data_asset_paths
