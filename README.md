# Chatbot with FastAPI and Supabase

This project is a Python-based chatbot that uses FastAPI for the web framework and Supabase as a backend database to retrieve information from Reddit posts. It also integrates with OpenAI for generating AI-powered responses.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Setup and Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    -   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file with the following content:
    ```
    fastapi
    uvicorn[standard]
    supabase
    python-dotenv
    openai
    numpy
    pillow
    pytesseract
    python-multipart
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project. This file will store your sensitive credentials. Add the following variables, replacing the placeholder values with your actual keys:

    ```env
    SUPABASE_URL=your_supabase_url_here
    SUPABASE_KEY=your_supabase_anon_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    *   `SUPABASE_URL`: Your Supabase project URL.
    *   `SUPABASE_KEY`: Your Supabase project anon (public) key.
    *   `OPENAI_API_KEY`: Your OpenAI API key for accessing models like GPT-3.5-turbo.

    **Important:** Ensure the `.env` file is listed in your `.gitignore` file to prevent committing sensitive keys to your repository.

## Running the Application

Once the dependencies are installed and the environment variables are set up, you can run the FastAPI application using Uvicorn:

```bash
uvicorn chatbot:app --reload
```

-   `chatbot:app`: `chatbot` refers to the Python file (`chatbot.py`) and `app` is the FastAPI instance within that file.
-   `--reload`: This flag enables auto-reloading, so the server will restart automatically when you make changes to the code.

The application will typically be available at `http://127.0.0.1:8000`. You can open this URL in your web browser to interact with the chatbot.

## Project Structure

-   `chatbot.py`: Main application file containing FastAPI routes, Supabase client initialization, OpenAI client initialization, and chatbot logic.
-   `index.html`: The HTML file for the chatbot's frontend interface.
-   `requirements.txt`: Lists the Python dependencies for the project.
-   `.env`: Stores environment variables (Supabase URL/Key, OpenAI API Key). **Should be in `.gitignore`**.
-   `README.md`: This file.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

<<<<<<< HEAD
=======
## Supabase Full-Text Search (FTS) Setup

For the Reddit post searching functionality to work effectively, ensure you have set up Full-Text Search on your `posts` table in Supabase. This typically involves:

1.  Creating a `tsvector` column (e.g., `fts_document`) in your `posts` table.
2.  Populating this column with data from the columns you want to search (e.g., `title` and `text`).
3.  Creating an index on this `tsvector` column.

Refer to the Supabase documentation for detailed instructions on setting up FTS. The application currently uses `fts_document` as the tsvector column name for searches. 
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3
