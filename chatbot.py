import os
from supabase import create_client, Client
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from dotenv import load_dotenv # Added for .env support
import openai # Added OpenAI library

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CHATBOT ROLE AND GUIDELINES (Now also for LLM):
# Role: AI Mortgage Discussion Assistant
# You are a helpful AI assistant. You can discuss general mortgage topics and 
# search a database of Reddit discussions (from r/firsttimehomebuyer) for related posts.
# When answering based on Reddit posts, state that the info is from a public forum and not financial advice.
# Be conversational and helpful.

# Initialize FastAPI app
app = FastAPI()

# --- Serve HTML Frontend ---
# Option 1: Simple HTMLResponse (if index.html is very basic and doesn't need templating)
# Option 2: Jinja2Templates (more flexible if you plan to pass data to HTML or use templates)
# We'll use Jinja2Templates as it's a common pattern and good practice.

# Assuming your index.html is in the same directory as chatbot.py
# If you create a 'static' folder for CSS/JS later, you'd mount it like this:
# app.mount("/static", StaticFiles(directory="static"), name="static")

# For serving index.html from the root directory
# Create a 'templates' directory and move 'index.html' into it for Jinja2, OR
# serve it directly if it's simple. For now, let's assume index.html is in the root.
# For simplicity in this step, we'll read it directly. A 'templates' dir is better for larger apps.

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Make sure index.html is in the same directory as chatbot.py.</p>", status_code=404)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(content=f"<h1>Internal Server Error</h1><p>Error serving HTML: {e}</p>", status_code=500)

# Initialize Supabase client from .env variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")
    supabase: Client | None = None
else:
    try:
        supabase: Client | None = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Supabase client: {e}")
        supabase: Client | None = None

# Initialize OpenAI Client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in .env file. AI model responses will be disabled.")
    ai_client = None
else:
    try:
        ai_client = openai.OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        ai_client = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Basic pre-defined mortgage Q&A
GENERAL_MORTGAGE_QA = {
    "what is a mortgage?": "A mortgage is a loan from a bank or other financial institution that helps a borrower purchase a home. The home is used as collateral for the loan.",
    "what is interest rate?": "The interest rate is the percentage of the loan amount that a lender charges a borrower for the use of assets. For mortgages, it significantly impacts your monthly payments.",
    "what is a down payment?": "A down payment is the initial upfront portion of the total cost of a property, paid by the buyer at the time of purchase. The rest is typically financed through a mortgage.",
    "fixed vs variable rate": "A fixed-rate mortgage has an interest rate that stays the same for the entire loan term. A variable-rate (or adjustable-rate) mortgage has an interest rate that can change periodically, based on market conditions."
    # Add more general Q&As here
}

# Basic pre-defined information about the database structure
DATABASE_INFO_QA = {
    "testkey123xyz": "This is a unique test response for DATABASE_INFO_QA.",
    "what database tables can you access?": "I can primarily access and search a table called 'posts', which contains discussions and content scraped from the r/firsttimehomebuyer subreddit. There are also related tables for 'comments' and 'attachments' for that content.",
    "what information is in the posts table?": "The 'posts' table stores information from Reddit posts, including the post title, the main text content (body), the author, a link to the original post (URL), and the Reddit ID. I use this to search for discussions on various mortgage-related topics.",
    "tell me about the posts table": "The 'posts' table is my main source of information for answering specific search queries. It includes fields like 'id', 'reddit_id', 'title', 'text', 'author', 'created_utc', 'url', and 'original_url'. I primarily use the 'title' and 'text' for searching.",
    "what is in the comments table?": "There's a 'comments' table designed to store comments related to the posts. It includes the comment body, author, and a link to its parent post.",
    "what is in the attachments table?": "An 'attachments' table exists to store links or references to any attachments associated with the posts, such as images or documents.",
    "what kind of data do you have?": "My data primarily consists of discussions, questions, and shared experiences related to first-time home buying and mortgages, sourced from the r/firsttimehomebuyer subreddit. This is stored in a 'posts' table and related 'comments' and 'attachments' tables."
    # Add more questions about the database structure here
}

async def get_ai_response(user_question: str, context_posts_text: str | None = None) -> str:
    if not ai_client:
        return "I am currently unable to connect to the AI model to generate a response. Please check your OPENAI_API_KEY in the .env file and Uvicorn startup logs for errors."

    system_prompt = (
        "You are an AI Mortgage Discussion Assistant. Your primary goal is to be informative, helpful, and clear. "
        "You have access to two main types of information to help you answer questions:\n"
        "1. Your general knowledge about mortgages, loans, and housing finance.\n"
        "2. Specific context that may be provided to you, derived from a database of discussions and document details shared by users on the r/firsttimehomebuyer subreddit. This data is NOT official, verified, or universally representative financial data; it reflects individual experiences and posts."
        "\n"
        "When answering:\n"
        "- For general mortgage questions (e.g., 'What is an FHA loan?'), use your broad knowledge."
        "- If you are given specific context from the subreddit data (it will be clearly marked as such in the prompt), use that context to answer. ALWAYS explicitly state that this information is based on user-shared content from the r/firsttimehomebuyer subreddit and should not be considered financial advice or official market data. For example, say 'Based on discussions from r/firsttimehomebuyer...' or 'Some users on the subreddit have mentioned...'"
        "- If asked to generalize, summarize, or identify trends (like average costs, common rates) based on provided subreddit data, do so cautiously. Preface your answer by stating the source and its informal nature. For example: 'Based on an analysis of N posts from r/firsttimehomebuyer, a common theme regarding X is Y. Remember, this is based on user-submitted content.'"
        "- If you are provided with structured data (e.g., a list of interest rates, loan amounts), you can perform simple calculations like averages if asked, but again, heavily caveat the source and reliability."
        "- DO NOT present information from the subreddit as absolute fact or as generally applicable financial guidance."
        "- DO NOT make up information, especially financial figures or specific document details, if it's not in your general knowledge or the provided context."
        "- If you cannot answer a question based on your knowledge or the provided context, say so clearly."
        "- Prioritize safety and privacy. Do not repeat or allude to any personally identifiable information (names, specific addresses, exact unique loan numbers) if it were to accidentally appear in the context provided to you. Focus on anonymized trends and general discussion points."
        "- Be conversational and aim to educate users about mortgages in simple terms."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    if context_posts_text:
        # Add context from Supabase search to the prompt for the LLM
        # This is a simple way; more advanced RAG would involve formatting and possibly chunking.
        context_message = f"Here is some context from r/firsttimehomebuyer that might be relevant to the user\'s question:\n\n---\n{context_posts_text}\n---\n\nPlease use this context to inform your answer if applicable."
        # Insert context message before the user's final question for better flow
        messages.insert(1, {"role": "system", "content": context_message}) # or assistant, depending on how you want to frame it

    try:
        logger.info(f"Sending request to AI model. User question: '{user_question}', Context provided: {bool(context_posts_text)}")
        completion = ai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" or other models you have access to
            messages=messages,
            temperature=0.7, # Controls randomness: lower is more deterministic
            max_tokens=300  # Adjust as needed for response length
        )
        response_content = completion.choices[0].message.content.strip()
        return response_content
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI API connection error: {e}")
        return "I'm having trouble connecting to the AI service. Please try again later."
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded: {e}")
        return "I'm currently receiving too many requests. Please try again in a moment."
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API status error: {e}")
        return f"An error occurred with the AI service (Status: {e.status_code}). Details: {e.message}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting AI response: {e}")
        return "Sorry, an unexpected error occurred while trying to generate an AI response."

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    question_text = request.question # Original question text for AI
    question_normalized = question_text.lower().strip().replace("?","") # Normalized for dictionary lookups
    response_message = ""
    context_for_ai = None

    # 1. Check for basic, pre-defined mortgage questions
    if question_normalized in GENERAL_MORTGAGE_QA:
        response_message = GENERAL_MORTGAGE_QA[question_normalized]
    
    # 2. Check for questions about the database itself
    if not response_message and question_normalized in DATABASE_INFO_QA:
        response_message = DATABASE_INFO_QA[question_normalized]

    # 3. If not a general or DB info QA, decide if we need to search Supabase or go directly to AI
    if not response_message: 
        # Simplified logic: if it's a search-like query, search Supabase first
        search_triggers = ["search posts for", "find posts about", "what do posts say about", "what are people saying about", "reddit search", "subreddit search"]
        is_search_query = any(trigger in question_normalized for trigger in search_triggers)
        # Or if the question implies wanting context from reddit for the AI to process
        reddit_context_triggers = ["from posts", "from reddit", "on the subreddit"]
        if not is_search_query and any(ctx_trigger in question_normalized for ctx_trigger in reddit_context_triggers):
            is_search_query = True # Treat as a search to gather context for AI

        if is_search_query:
            if not supabase:
                response_message = "I am unable to connect to the database to search for specific posts. Please check server configuration."
            else:
                search_term = question_text # Use original question text or a parsed version for search term
                # Basic parsing for search term (can be improved)
                for trigger in search_triggers + reddit_context_triggers:
                    if trigger in question_normalized:
                        search_term = question_text.lower().split(trigger, 1)[-1].strip().replace("?","")
                        break
                if not search_term: search_term = question_text # Fallback to full question if parsing fails

                try:
                    logger.info(f"Searching 'posts' table for context for AI: '{search_term}'")
                    query = supabase.table("posts").select("title, text").text_search("fts_document", f"'{search_term}'").limit(3).execute()
                    
                    if query.data:
                        # Format context for the AI
                        context_for_ai = "\n".join([f"Title: {p.get('title', '')}\nText: {p.get('text', '')[:500]}..." for p in query.data]) # Limit text length per post
                        # For this flow, we won't send Supabase results directly to user, but to AI
                        # response_message = "I found some Reddit posts. I will summarize them with AI."
                        # Instead, we now pass context_for_ai to the get_ai_response function
                        logger.info(f"Context found for AI: {len(query.data)} posts.")
                    else:
                        logger.info(f"No direct context found in Supabase for '{search_term}'. Will ask AI without specific Reddit context.")
                        # response_message = f"I couldn't find specific Reddit posts for '{search_term}'. I can try asking the AI generally."
                except Exception as e:
                    logger.error(f"Error querying 'posts' table for AI context: {e}")
                    # response_message = "Error getting context from database. Asking AI generally."
        
        # 4. Always call the AI model if no predefined answer was found
        # It will use context_for_ai if it was populated by Supabase search
        response_message = await get_ai_response(question_text, context_posts_text=context_for_ai)
        
    # 5. Fallback if AI somehow failed and response_message is still empty (should be handled by get_ai_response)
    if not response_message:
        response_message = "I'm having a bit of trouble thinking right now. Please try asking again."

    return QueryResponse(answer=response_message)

# To run this app:
# 1. Create a .env file in this directory with your SUPABASE_URL and SUPABASE_KEY.
#    Example .env file content:
#    SUPABASE_URL=your_supabase_url_here
#    SUPABASE_KEY=your_supabase_anon_key_here
# 2. Ensure Full-Text Search (FTS) is set up on your 'posts' table in Supabase,
#    using a tsvector column named 'fts_document'. (See previous instructions for FTS setup).
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run the app with: uvicorn chatbot:app --reload 