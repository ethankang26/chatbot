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

# CHATBOT ROLE AND GUIDELINES:
# Role: AI Mortgage Information Assistant
# You are a helpful AI assistant that can discuss general mortgage topics and 
# search through extracted text from mortgage-related images posted on r/firsttimehomebuyer.
# This includes screenshots of loan documents, mortgage statements, closing documents, etc.
# When answering based on extracted text from Reddit images, state that the info is from 
# user-shared content and not financial advice.

# Initialize FastAPI app
app = FastAPI()

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
    "what database tables can you access?": "I can access a 'posts' table with Reddit discussions from r/firsttimehomebuyer, a 'comments' table for Reddit comments, and most importantly, an 'attachments' table containing extracted text from images (like mortgage documents, loan papers, etc.) posted on the subreddit.",
    "what information is in the posts table?": "The 'posts' table stores information from Reddit posts, including the post title, the main text content (body), the author, a link to the original post (URL), and the Reddit ID. I use this to search for discussions on various mortgage-related topics.",
    "tell me about the posts table": "The 'posts' table is my source for Reddit discussion data. It includes fields like 'id', 'reddit_id', 'title', 'text', 'author', 'created_utc', 'url', and 'original_url'. I primarily use the 'title' and 'text' for searching.",
    "what is in the comments table?": "The 'comments' table stores comments from Reddit posts, including the comment body, author, and a link to its parent post.",
    "what is in the attachments table?": "The 'attachments' table contains extracted text from images posted on r/firsttimehomebuyer. This includes OCR'd text from screenshots of mortgage documents, loan papers, closing statements, rate sheets, and other mortgage-related photos that users share on Reddit.",
    "what kind of data do you have?": "I have access to Reddit discussions from r/firsttimehomebuyer, including text extracted from images of mortgage documents, loan papers, and other financial documents that users share in their posts. This data helps me answer questions about mortgage experiences and document details shared by the community."
}

# Information about the attachments table and extracted text from Reddit images
ATTACHMENTS_INFO_QA = {
    "what is in the attachments table?": "The 'attachments' table contains text extracted from images posted on r/firsttimehomebuyer. This includes OCR'd text from mortgage documents, loan papers, closing statements, and other financial documents that Reddit users share as screenshots or photos.",
    "what kind of attachments do you have?": "I have access to text extracted from various mortgage-related images posted on r/firsttimehomebuyer, such as loan estimates, closing disclosures, rate lock documents, pre-approval letters, and mortgage statements that users share for advice or discussion.",
    "can you search the extracted text from reddit images?": "Yes! I can search through text extracted from mortgage-related images posted on r/firsttimehomebuyer. This includes information from loan documents, rate sheets, closing papers, and other financial documents that users have shared as photos or screenshots.",
    "how do you get the extracted text?": "The text is extracted from images using OCR (Optical Character Recognition) technology when users post photos of their mortgage documents on r/firsttimehomebuyer. This extracted text is then stored in our database for searching and analysis."
}

# Enhanced get_ai_response function for handling extracted text from Reddit images
async def get_ai_response(user_question: str, context_posts_text: str | None = None) -> str:
    if not ai_client:
        return "I am currently unable to connect to the AI model to generate a response. Please check your OPENAI_API_KEY in the .env file and Uvicorn startup logs for errors."

    system_prompt = (
        "You are an AI Mortgage Information Assistant that helps users understand mortgage-related information. "
        "You have access to two main sources of information:\n\n"
        
        "1. Your general knowledge about mortgages, loans, and housing finance\n"
        "2. Text extracted from mortgage-related images posted on r/firsttimehomebuyer subreddit\n\n"
        
        "The extracted text comes from OCR processing of photos and screenshots of:\n"
        "- Loan estimates and closing disclosures\n"
        "- Mortgage statements and payment schedules\n"
        "- Rate lock documents and pre-approval letters\n"
        "- Other mortgage-related documents shared by Reddit users\n\n"
        
        "When answering:\n"
        "- If using extracted text from Reddit images, always clarify the source by saying something like 'Based on mortgage documents shared on r/firsttimehomebuyer...' or 'From Reddit posts showing mortgage paperwork...'\n"
        "- Remember this is user-shared content from a public forum, not official financial advice\n"
        "- Help explain mortgage terms and numbers found in the extracted text in simple language\n"
        "- If you see specific rates, fees, or terms in the extracted text, highlight and explain them\n"
        "- For general mortgage questions without specific document context, use your knowledge\n"
        "- Be helpful in interpreting mortgage jargon and explaining what documents mean\n"
        "- Always remind users that information from Reddit posts should be verified with their own lenders\n\n"
        
        "Be conversational, educational, and focus on helping users understand mortgage concepts and document details."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    if context_posts_text:
        # Determine the type of context
        if "From Reddit image" in context_posts_text or "extracted from image" in context_posts_text.lower():
            context_label = "text extracted from mortgage-related images posted on r/firsttimehomebuyer"
        elif "Reddit Post" in context_posts_text:
            context_label = "discussions from r/firsttimehomebuyer subreddit"
        else:
            context_label = "relevant mortgage information from Reddit"
        
        context_message = f"Here is {context_label} that may help answer the user's question:\n\n---\n{context_posts_text}\n---\n\nPlease use this information to provide a helpful answer, remembering to cite the source appropriately."
        messages.insert(1, {"role": "system", "content": context_message})

    try:
        logger.info(f"Sending request to AI model. User question: '{user_question}', Context type: {'Extracted Image Text' if context_posts_text and 'image' in context_posts_text.lower() else 'General' if context_posts_text else 'None'}")
        completion = ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=400
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

# Function to search extracted text from Reddit images
async def search_extracted_text_from_images(search_term: str, limit: int = 5) -> str | None:
    """Search the attachments table for extracted text from Reddit images"""
    if not supabase:
        return None
    
    try:
        logger.info(f"Searching extracted text from Reddit images for: '{search_term}'")
        
        # Search using ilike for partial text matching in extracted_text
        query = supabase.table("attachments").select(
            "id, extracted_text, post_id, created_at"
        ).ilike("extracted_text", f"%{search_term}%").limit(limit).execute()
        
        if query.data:
            # Format context for the AI
            context_parts = []
            for attachment in query.data:
                post_id = attachment.get('post_id', 'Unknown post')
                text = attachment.get('extracted_text', '')
                # Extract relevant excerpt around the search term
                excerpt = extract_relevant_excerpt(text, search_term, max_length=400)
                context_parts.append(f"From Reddit image (Post ID: {post_id}):\n{excerpt}")
            
            context_for_ai = "\n\n".join(context_parts)
            logger.info(f"Found {len(query.data)} relevant extracted texts from Reddit images")
            return context_for_ai
        else:
            logger.info(f"No extracted text found for '{search_term}'")
            return None
            
    except Exception as e:
        logger.error(f"Error searching extracted text from images: {e}")
        return None

# Function to extract relevant excerpts from text
def extract_relevant_excerpt(text: str, search_term: str, max_length: int = 400) -> str:
    """Extract relevant excerpt from text around the search term"""
    if not text:
        return "No text available."
    
    # Convert to lowercase for case-insensitive search
    text_lower = text.lower()
    search_lower = search_term.lower()
    
    # Find the position of search term in the text
    search_words = search_lower.split()
    
    for word in search_words:
        index = text_lower.find(word)
        if index != -1:
            # Extract context around the found word
            start = max(0, index - 150)
            end = min(len(text), index + max_length - 150)
            excerpt = text[start:end]
            
            # Clean up the excerpt
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
                
            return excerpt
    
    # If no specific match, return beginning of text
    return text[:max_length] + ("..." if len(text) > max_length else "")

# Function to search for mortgage-specific information in extracted image text
async def search_mortgage_info_in_images(search_term: str) -> str | None:
    """Search for mortgage-specific information in extracted text from Reddit images"""
    if not supabase:
        return None
    
    try:
        logger.info(f"Searching for mortgage info in Reddit images: '{search_term}'")
        
        # Search for mortgage-related content with the user's search term
        query = supabase.table("attachments").select(
            "id, extracted_text, post_id, created_at"
        ).or_(
            f"extracted_text.ilike.%{search_term}%,"
            f"extracted_text.ilike.%mortgage%,"
            f"extracted_text.ilike.%loan%,"
            f"extracted_text.ilike.%rate%,"
            f"extracted_text.ilike.%apr%,"
            f"extracted_text.ilike.%payment%"
        ).limit(5).execute()
        
        if query.data:
            context_parts = []
            for attachment in query.data:
                post_id = attachment.get('post_id', 'Unknown post')
                text = attachment.get('extracted_text', '')
                excerpt = extract_relevant_excerpt(text, search_term, max_length=500)
                context_parts.append(f"From Reddit image extracted from mortgage document (Post ID: {post_id}):\n{excerpt}")
            
            context_for_ai = "\n\n".join(context_parts)
            logger.info(f"Found {len(query.data)} mortgage-related extracted texts")
            return context_for_ai
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error searching mortgage info in extracted texts: {e}")
        return None

# Enhanced ask_question function with support for extracted text from Reddit images
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    question_text = request.question
    question_normalized = question_text.lower().strip().replace("?", "")
    response_message = ""
    context_for_ai = None

    # 1. Check for basic, pre-defined mortgage questions
    if question_normalized in GENERAL_MORTGAGE_QA:
        response_message = GENERAL_MORTGAGE_QA[question_normalized]
    
    # 2. Check for questions about the database itself
    if not response_message and question_normalized in DATABASE_INFO_QA:
        response_message = DATABASE_INFO_QA[question_normalized]
    
    # 3. Check for questions about attachments/extracted text
    if not response_message and question_normalized in ATTACHMENTS_INFO_QA:
        response_message = ATTACHMENTS_INFO_QA[question_normalized]

    # 4. If not a predefined answer, search extracted text and use AI
    if not response_message:
        # Check if this is asking about images, documents, or extracted text
        image_triggers = [
            "image", "photo", "screenshot", "picture", "document", "extracted text",
            "ocr", "attachment", "mortgage document", "loan document"
        ]
        
        # Check if it's a mortgage-related query
        mortgage_triggers = [
            "mortgage", "loan", "interest", "rate", "payment", "closing",
            "down payment", "refinance", "apr", "principal", "escrow", "pmi"
        ]
        
        is_image_query = any(trigger in question_normalized for trigger in image_triggers)
        is_mortgage_query = any(trigger in question_normalized for trigger in mortgage_triggers)
        
        # Search strategy based on query type
        if is_image_query or is_mortgage_query:
            # Search in extracted text from Reddit images
            context_for_ai = await search_mortgage_info_in_images(question_text)
            
            if not context_for_ai:
                # Fallback to general extracted text search
                context_for_ai = await search_extracted_text_from_images(question_text)
        else:
            # For general questions, still check extracted text first
            context_for_ai = await search_extracted_text_from_images(question_text)
            
            # If no extracted text context found, try the Reddit posts table as fallback
            if not context_for_ai and supabase:
                try:
                    query = supabase.table("posts").select("title, text").text_search("fts_document", f"'{question_text}'").limit(3).execute()
                    if query.data:
                        context_for_ai = "\n".join([f"Reddit Post - Title: {p.get('title', '')}\nText: {p.get('text', '')[:300]}..." for p in query.data])
                except Exception as e:
                    logger.error(f"Error searching posts table: {e}")

        # Always get AI response with or without context
        if supabase is None:
            response_message = "I am unable to connect to the database to search for extracted text from images. I can provide general mortgage information though."
        
        # Generate AI response with context if available
        response_message = await get_ai_response(question_text, context_posts_text=context_for_ai)

    # 5. Fallback if somehow still empty
    if not response_message:
        response_message = "I'm having a bit of trouble thinking right now. Please try asking again."

    return QueryResponse(answer=response_message)

# To run this app:
# 1. Create a .env file in this directory with your SUPABASE_URL and SUPABASE_KEY.
#    Example .env file content:
#    SUPABASE_URL=your_supabase_url_here
#    SUPABASE_KEY=your_supabase_anon_key_here
#    OPENAI_API_KEY=your_openai_api_key_here
# 2. Ensure your 'attachments' table has an 'extracted_text' column with OCR'd text from Reddit images
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run the app with: uvicorn chatbot:app --reload