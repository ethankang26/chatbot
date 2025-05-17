import os
<<<<<<< HEAD
import base64
from supabase import create_client, Client
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Depends, Cookie, Response
from fastapi.responses import HTMLResponse, JSONResponse
=======
from supabase import create_client, Client
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
<<<<<<< HEAD
from dotenv import load_dotenv
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Union
import uuid
import shutil
import tempfile
from PIL import Image
import pytesseract
=======
from dotenv import load_dotenv # Added for .env support
import openai # Added OpenAI library
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
# Initialize FastAPI app
app = FastAPI(title="Mortgage Information Assistant")

# Create temporary directory for image uploads
TEMP_DIR = tempfile.gettempdir()

# Store conversation history (in memory for simplicity)
conversation_histories: Dict[str, List[Dict[str, str]]] = {}
=======
# CHATBOT ROLE AND GUIDELINES:
# Role: AI Mortgage Information Assistant
# You are a helpful AI assistant that can discuss general mortgage topics and 
# search through Reddit posts, comments, and extracted text from mortgage-related images 
# posted on r/firsttimehomebuyer. All data is linked by post_id.

# Initialize FastAPI app
app = FastAPI()
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
<<<<<<< HEAD
        with open("index.html", "r", encoding="utf-8") as f:
=======
        with open("index.html", "r") as f:
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Make sure index.html is in the same directory as chatbot.py.</p>", status_code=404)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(content=f"<h1>Internal Server Error</h1><p>Error serving HTML: {e}</p>", status_code=500)

# Initialize Supabase client from .env variables
supabase_url = os.getenv("SUPABASE_URL")
<<<<<<< HEAD
supabase_key = os.getenv("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    logger.error("SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file.")
    supabase: Optional[Client] = None
else:
    try:
        supabase: Optional[Client] = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Supabase client: {e}")
        supabase: Optional[Client] = None
=======
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
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3

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

<<<<<<< HEAD
# Define request and response models
class QueryRequest(BaseModel):
    question: str
    image_data: Optional[str] = None
=======
class QueryRequest(BaseModel):
    question: str
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3

class QueryResponse(BaseModel):
    answer: str

<<<<<<< HEAD
class RetrievedContext(BaseModel):
    source: str
    content: str
    post_id: Optional[int] = None

class ImageUploadResponse(BaseModel):
    success: bool
    image_data: str = ""
    message: str = ""

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4.1-mini"  # Changed to a vision-capable model
MAX_TOKENS = 500
EMBEDDING_DIMENSION = 1536  # Dimension of text-embedding-3-small
SIMILARITY_THRESHOLD = 0
MAX_RESULTS_PER_TABLE = 5
MAX_CONTEXT_LENGTH = 3000

# Create embeddings for user query
async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding vector for text using OpenAI API"""
    if not ai_client:
        logger.error("OpenAI client not available for embedding generation")
        return None
    
    try:
        text = text.replace("\n", " ")
        embedding_response = ai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        embedding = embedding_response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# Function to search across all tables using vector similarity
async def vector_search(query_embedding: List[float], limit_per_table: int = MAX_RESULTS_PER_TABLE) -> List[RetrievedContext]:
    """Search all tables using vector similarity with the embeddings"""
    if not supabase or not query_embedding:
        return []
    
    results = []
    
    try:
        # 1. Search Posts table
        posts_rpc = supabase.rpc(
            "match_posts_embeddings", 
            {"query_embedding": query_embedding, "match_threshold": SIMILARITY_THRESHOLD, "match_count": limit_per_table}
        ).execute()
        
        if posts_rpc.data:
            print("Post similarity scores:")
            for post in posts_rpc.data:
                post_id = post.get('id')
                text = post.get('text', '')
                title = post.get('title', 'Reddit Post')
                similarity = post.get('similarity', 'N/A')  # Extract similarity score
                print(f"Post {post_id}: {similarity}")
                results.append(RetrievedContext(
                    source=f"Reddit Post: {title}",
                    content=text,
                    post_id=post_id
                ))
        
        # 2. Search Comments table
        comments_rpc = supabase.rpc(
            "match_comments_embeddings", 
            {"query_embedding": query_embedding, "match_threshold": SIMILARITY_THRESHOLD, "match_count": limit_per_table}
        ).execute()
        
        if comments_rpc.data:
            print("Comment similarity scores:")
            for comment in comments_rpc.data:
                post_id = comment.get('post_id')
                body = comment.get('body', '')
                similarity = comment.get('similarity', 'N/A')  # Extract similarity score
                print(f"Comment for Post {post_id}: {similarity}")
                results.append(RetrievedContext(
                    source=f"Comment on Post {post_id}",
                    content=body,
                    post_id=post_id
                ))
        
        # 3. Search Attachments table (extracted text from images)
        attachments_rpc = supabase.rpc(
            "match_attachments_embeddings", 
            {"query_embedding": query_embedding, "match_threshold": SIMILARITY_THRESHOLD, "match_count": limit_per_table}
        ).execute()
        
        if attachments_rpc.data:
            print("Attachment similarity scores:")
            for attachment in attachments_rpc.data:
                post_id = attachment.get('post_id')
                text = attachment.get('extracted_text', '')
                similarity = attachment.get('similarity', 'N/A')  # Extract similarity score
                print(f"Attachment for Post {post_id}: {similarity}")
                results.append(RetrievedContext(
                    source=f"Document from Post {post_id}",
                    content=text,
                    post_id=post_id
                ))
        print(results,"nopoppop")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []

# Enhanced get_ai_response function for RAG with vision capabilities
async def get_ai_response(
    user_question: str, 
    context: List[RetrievedContext], 
    image_data: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Generate AI response with retrieved context, optional image, and conversation history"""
    if not ai_client:
        return "I am currently unable to connect to the AI model to generate a response. Please check your OPENAI_API_KEY in the .env file."
=======
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
    "what database tables can you access?": "I can access three related tables: 'posts' (Reddit post text and IDs), 'comments' (comment bodies linked to posts via post_id), and 'attachments' (extracted text from images linked to posts via post_id). All tables are connected by post_id for comprehensive search.",
    "what information is in the posts table?": "The 'posts' table contains Reddit post text and IDs from r/firsttimehomebuyer. I primarily search the 'text' column for post content and use 'id' to link with comments and attachments.",
    "tell me about the posts table": "The 'posts' table stores the main Reddit post content. Key columns I use are 'text' (the post content) and 'id' (which becomes the post_id in related tables).",
    "what is in the comments table?": "The 'comments' table stores Reddit comment bodies with 'post_id' linking them back to the original posts. I search the 'body' column for comment content.",
    "what is in the attachments table?": "The 'attachments' table contains extracted text from images (OCR'd mortgage documents, loan papers, etc.) with 'post_id' linking them to the original Reddit posts that contained these images.",
    "what kind of data do you have?": "I have comprehensive mortgage-related data from r/firsttimehomebuyer: original post text, user comments, and text extracted from mortgage document images. All data is linked by post_id, so I can provide context from the full discussion thread including any attached documents.",
    "how are the tables connected?": "All tables are connected by post_id: Posts have an 'id', Comments have 'post_id' referencing the post, and Attachments have 'post_id' referencing the post. This allows me to find related discussions and documents for any topic."
}

# Information about the attachments table and extracted text from Reddit images
ATTACHMENTS_INFO_QA = {
    "what is in the attachments table?": "The 'attachments' table contains text extracted from mortgage-related images posted on r/firsttimehomebuyer. Each attachment has a 'post_id' linking it back to the original Reddit post, allowing me to provide full context from the discussion.",
    "what kind of attachments do you have?": "I have text extracted from various mortgage documents shared as images on Reddit: loan estimates, closing disclosures, rate sheets, pre-approval letters, and mortgage statements. Each is linked to its original post discussion.",
    "can you search the extracted text from reddit images?": "Yes! I can search through extracted text from mortgage images and also show you the related post and comments for full context, since everything is linked by post_id.",
    "how do you get the extracted text?": "The text is extracted from images using OCR technology when users post photos of mortgage documents on r/firsttimehomebuyer. This extracted text is stored with the post_id to maintain connection to the original discussion."
}

# Enhanced get_ai_response function for handling all three data sources
async def get_ai_response(user_question: str, context_posts_text: str | None = None) -> str:
    if not ai_client:
        return "I am currently unable to connect to the AI model to generate a response. Please check your OPENAI_API_KEY in the .env file and Uvicorn startup logs for errors."
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3

    system_prompt = (
        "You are an AI Mortgage Information Assistant. Provide concise, helpful answers about mortgage topics.\n\n"
        
        "You have access to:\n"
        "1. General mortgage knowledge\n"
        "2. Reddit posts from r/firsttimehomebuyer\n"
        "3. Community comments and discussions\n"
<<<<<<< HEAD
        "4. Images shared by the user\n\n"
        "5. data from the database\n\n"
=======
        "4. Extracted text from mortgage documents shared as images\n\n"
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3
        
        "Response guidelines:\n"
        "- Keep answers brief and to the point (2-3 paragraphs max)\n"
        "- Only include the most relevant information\n"
<<<<<<< HEAD
        "- Mention sources briefly: 'Based on Reddit posts...' or 'From the image shared...'\n"
=======
        "- Mention sources briefly: 'Based on Reddit posts...' or 'From mortgage documents shared...'\n"
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3
        "- Focus on key numbers or insights, not exhaustive details\n"
        "- If the user wants more detail, they can ask follow-up questions\n"
        "- Remember this is community-shared content, not official financial advice\n\n"
        
        "Be conversational and helpful, but keep it concise!"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
<<<<<<< HEAD
    ]

    # Add conversation history (if available)
    if conversation_history:
        # Only include up to the last 10 messages to avoid token limits
        for message in conversation_history[-10:]:
            if message["role"] != "system":  # Skip system messages in history
                messages.append(message)

    if context:
        # Format context for the AI
        context_text = ""
        for item in context:
            context_text += f"SOURCE: {item.source}\n{item.content}\n\n"
        
        # Limit context length
        if len(context_text) > MAX_CONTEXT_LENGTH:
            context_text = context_text[:MAX_CONTEXT_LENGTH] + "..."
        
        context_message = f"Here's relevant information from r/firsttimehomebuyer (provide a concise answer):\n\n{context_text}"
        messages.append({"role": "system", "content": context_message})
    
    # Add user question with image if provided
    if image_data and image_data.strip():
        # The message with image needs to be constructed as an array of content parts
        user_message_content = [
            {"type": "text", "text": user_question},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
        messages.append({"role": "user", "content": user_message_content})
    elif not conversation_history:
        # Only add the user question if it's not already in the conversation history
        messages.append({"role": "user", "content": user_question})

    try:
        logger.info(f"Sending request to AI model with conversation history. User question: '{user_question}'")
        completion = ai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=MAX_TOKENS
        )
        response_content = completion.choices[0].message.content.strip()
        return response_content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"I'm having trouble connecting to the AI service: {str(e)[:100]}. Please try again later."

# Function to get database statistics
async def get_database_stats() -> Dict[str, int]:
    """Get real-time database statistics"""
    if not supabase:
        return {"posts": 0, "comments": 0, "attachments": 0}
    
    stats = {}
    try:
        # Get post count
        posts_result = supabase.table("posts").select("id", count="exact").execute()
        stats["posts"] = posts_result.count if hasattr(posts_result, 'count') else 0
        
        # Get comment count
        comments_result = supabase.table("comments").select("post_id", count="exact").execute()
        stats["comments"] = comments_result.count if hasattr(comments_result, 'count') else 0
        
        # Get attachment count
        attachments_result = supabase.table("attachments").select("post_id", count="exact").execute()
        stats["attachments"] = attachments_result.count if hasattr(attachments_result, 'count') else 0
        
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"posts": 0, "comments": 0, "attachments": 0}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Process an uploaded image and return base64 encoded data"""
    # Check file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = ['.png', '.jpg', '.jpeg']
    
    if file_extension not in allowed_extensions:
        return ImageUploadResponse(
            success=False,
            message=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read the file content
        file_content = await file.read()
        
        # Convert to base64
        base64_encoded = base64.b64encode(file_content).decode('utf-8')
        data_url = f"data:image/{file_extension.replace('.', '')};base64,{base64_encoded}"
        
        return ImageUploadResponse(
            success=True,
            image_data=data_url,
            message="Image processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        return ImageUploadResponse(
            success=False,
            message=f"Error processing image: {str(e)}"
        )

@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    # Generate a session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
    
    # Get or create conversation history for this session
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    question_text = request.question
    image_data = request.image_data
    
    if not question_text or question_text.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Add user message to history
    conversation_histories[session_id].append({"role": "user", "content": question_text})
    
    # Check for database statistics requests
    if any(phrase in question_text.lower() for phrase in ["database stats", "how many posts", "how many comments", "how many attachments"]):
        stats = await get_database_stats()
        stats_response = (
            f"Current database statistics:\n"
            f"• Posts: {stats['posts']}\n"
            f"• Comments: {stats['comments']}\n"
            f"• Attachments: {stats['attachments']}"
        )
        # Add response to history
        conversation_histories[session_id].append({"role": "assistant", "content": stats_response})
        return QueryResponse(answer=stats_response)
    
    # Generate embedding for the question
    query_embedding = await generate_embedding(question_text)
    
    if not query_embedding:
        # Fallback to general response if embedding generation fails
        fallback_response = "I'm having trouble processing your question semantically. I can still provide general mortgage information, but personalized results are currently unavailable."
        # Add response to history
        conversation_histories[session_id].append({"role": "assistant", "content": fallback_response})
        return QueryResponse(answer=fallback_response)
    
    # Perform vector search using embeddings
    search_results = await vector_search(query_embedding)
    
    # Generate AI response with retrieved context, image, and conversation history
    response_message = await get_ai_response(question_text, search_results, image_data, conversation_histories[session_id])
    
    # Add response to history
    conversation_histories[session_id].append({"role": "assistant", "content": response_message})
    
    return QueryResponse(answer=response_message)

# Entry point for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)

# Note: For the vector similarity search to work, you need to create the following SQL functions in your Supabase database:
# 1. match_posts_embeddings(query_embedding vector, match_threshold float, match_count int)
# 2. match_comments_embeddings(query_embedding vector, match_threshold float, match_count int)
# 3. match_attachments_embeddings(query_embedding vector, match_threshold float, match_count int)
=======
        {"role": "user", "content": user_question}
    ]

    if context_posts_text:
        # Limit context length to prevent overwhelming responses
        if len(context_posts_text) > 1000:
            context_posts_text = context_posts_text[:1000] + "..."
        
        context_message = f"Here's relevant info from r/firsttimehomebuyer (keep response brief):\n\n{context_posts_text}\n\nProvide a concise answer focusing on key points only."
        messages.insert(1, {"role": "system", "content": context_message})

    try:
        logger.info(f"Sending request to AI model. User question: '{user_question}', Context provided: {bool(context_posts_text)}")
        completion = ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200  # Reduced from 500 to keep responses shorter
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

# Function to search across all three tables (posts, comments, attachments)
async def search_all_reddit_data(search_term: str, limit: int = 5) -> str | None:
    """Search posts, comments, and extracted text from images for comprehensive results"""
    if not supabase:
        return None
    
    try:
        logger.info(f"Searching all Reddit data for: '{search_term}'")
        
        context_parts = []
        
        # 1. Search Posts table (text column)
        try:
            posts_query = supabase.table("posts").select(
                "id, text"
            ).ilike("text", f"%{search_term}%").limit(limit).execute()
            
            if posts_query.data:
                for post in posts_query.data:
                    post_id = post.get('id')
                    text = post.get('text', '')
                    excerpt = extract_relevant_excerpt(text, search_term, max_length=300)
                    context_parts.append(f"Reddit Post (ID: {post_id}):\n{excerpt}")
                logger.info(f"Found {len(posts_query.data)} relevant posts")
        except Exception as e:
            logger.error(f"Error searching posts: {e}")
        
        # 2. Search Comments table (body column)
        try:
            comments_query = supabase.table("comments").select(
                "post_id, body"
            ).ilike("body", f"%{search_term}%").limit(limit).execute()
            
            if comments_query.data:
                for comment in comments_query.data:
                    post_id = comment.get('post_id')
                    body = comment.get('body', '')
                    excerpt = extract_relevant_excerpt(body, search_term, max_length=300)
                    context_parts.append(f"Comment on Post {post_id}:\n{excerpt}")
                logger.info(f"Found {len(comments_query.data)} relevant comments")
        except Exception as e:
            logger.error(f"Error searching comments: {e}")
        
        # 3. Search Attachments table (extracted_text column)
        try:
            attachments_query = supabase.table("attachments").select(
                "post_id, extracted_text"
            ).ilike("extracted_text", f"%{search_term}%").limit(limit).execute()
            
            if attachments_query.data:
                for attachment in attachments_query.data:
                    post_id = attachment.get('post_id')
                    text = attachment.get('extracted_text', '')
                    excerpt = extract_relevant_excerpt(text, search_term, max_length=400)
                    context_parts.append(f"Extracted from image in Post {post_id}:\n{excerpt}")
                logger.info(f"Found {len(attachments_query.data)} relevant extracted texts")
        except Exception as e:
            logger.error(f"Error searching attachments: {e}")
        
        # Combine all results
        if context_parts:
            context_for_ai = "\n\n".join(context_parts)
            logger.info(f"Total search results found: {len(context_parts)} items")
            return context_for_ai
        else:
            logger.info(f"No results found across all tables for '{search_term}'")
            return None
            
    except Exception as e:
        logger.error(f"Error in comprehensive search: {e}")
        return None

# Function to search for mortgage-specific information across all tables
async def search_mortgage_info_comprehensive(search_term: str) -> str | None:
    """Search for mortgage-specific information across posts, comments, and attachments"""
    if not supabase:
        return None
    
    try:
        logger.info(f"Comprehensive mortgage search for: '{search_term}'")
        
        # Define mortgage-related search conditions
        mortgage_conditions = [
            f"text.ilike.%{search_term}%",
            f"text.ilike.%mortgage%",
            f"text.ilike.%loan%",
            f"text.ilike.%rate%",
            f"text.ilike.%payment%"
        ]
        
        context_parts = []
        
        # 1. Search Posts with mortgage focus
        try:
            posts_query = supabase.table("posts").select(
                "id, text"
            ).or_(",".join(mortgage_conditions)).limit(3).execute()
            
            if posts_query.data:
                for post in posts_query.data:
                    post_id = post.get('id')
                    text = post.get('text', '')
                    excerpt = extract_relevant_excerpt(text, search_term, max_length=350)
                    context_parts.append(f"Reddit Post (ID: {post_id}):\n{excerpt}")
        except Exception as e:
            logger.error(f"Error in mortgage search - posts: {e}")
        
        # 2. Search Comments with mortgage focus
        try:
            comment_conditions = [condition.replace("text", "body") for condition in mortgage_conditions]
            comments_query = supabase.table("comments").select(
                "post_id, body"
            ).or_(",".join(comment_conditions)).limit(3).execute()
            
            if comments_query.data:
                for comment in comments_query.data:
                    post_id = comment.get('post_id')
                    body = comment.get('body', '')
                    excerpt = extract_relevant_excerpt(body, search_term, max_length=350)
                    context_parts.append(f"Comment on Post {post_id}:\n{excerpt}")
        except Exception as e:
            logger.error(f"Error in mortgage search - comments: {e}")
        
        # 3. Search Attachments with mortgage focus
        try:
            attachment_conditions = [condition.replace("text", "extracted_text") for condition in mortgage_conditions]
            attachments_query = supabase.table("attachments").select(
                "post_id, extracted_text"
            ).or_(",".join(attachment_conditions)).limit(3).execute()
            
            if attachments_query.data:
                for attachment in attachments_query.data:
                    post_id = attachment.get('post_id')
                    text = attachment.get('extracted_text', '')
                    excerpt = extract_relevant_excerpt(text, search_term, max_length=400)
                    context_parts.append(f"Extracted from mortgage document in Post {post_id}:\n{excerpt}")
        except Exception as e:
            logger.error(f"Error in mortgage search - attachments: {e}")
        
        if context_parts:
            context_for_ai = "\n\n".join(context_parts)
            logger.info(f"Mortgage search found: {len(context_parts)} relevant items")
            return context_for_ai
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in comprehensive mortgage search: {e}")
        return None

# Function to get dynamic database statistics
async def get_database_stats(stat_type: str) -> str:
    """Get real-time database statistics"""
    if not supabase:
        return "Unable to connect to database to get statistics."
    
    try:
        if stat_type == "post_count":
            result = supabase.table("posts").select("id", count="exact").execute()
            count = result.count
            return f"There are currently {count} posts in the database."
        elif stat_type == "comment_count":
            result = supabase.table("comments").select("post_id", count="exact").execute()
            count = result.count
            return f"There are currently {count} comments in the database."
        elif stat_type == "attachment_count":
            result = supabase.table("attachments").select("post_id", count="exact").execute()
            count = result.count
            return f"There are currently {count} attachments with extracted text in the database."
        else:
            return "Unknown statistic requested."
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return f"Error retrieving database statistics: {e}"
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

# Enhanced ask_question function with comprehensive search across all tables
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

    # 4. If not a predefined answer, check for database statistics requests
    if not response_message:
        # Check for database statistics queries
        if "how many posts" in question_normalized or "post count" in question_normalized:
            response_message = await get_database_stats("post_count")
        elif "how many comments" in question_normalized or "comment count" in question_normalized:
            response_message = await get_database_stats("comment_count")
        elif "how many attachments" in question_normalized or "attachment count" in question_normalized:
            response_message = await get_database_stats("attachment_count")

    # 5. If still not a predefined answer, search all tables and use AI
    # 5. If still not a predefined answer, search all tables and use AI
    if not response_message:
        # Check if it's a mortgage-related query
        mortgage_triggers = [
            "mortgage", "loan", "interest", "rate", "payment", "closing",
            "down payment", "refinance", "apr", "principal", "escrow", "pmi",
            "credit score", "lender", "preapproval", "closing costs"
        ]
        
        is_mortgage_query = any(trigger in question_normalized for trigger in mortgage_triggers)
        
        # Use comprehensive search strategy
        if is_mortgage_query:
            # Use mortgage-focused search across all tables
            context_for_ai = await search_mortgage_info_comprehensive(question_text)
        else:
            # Use general search across all tables
            context_for_ai = await search_all_reddit_data(question_text)

        # Always get AI response with or without context
        if supabase is None:
            response_message = "I am unable to connect to the database to search for information. I can provide general mortgage information though."
        else:
            # Generate AI response with context if available
            response_message = await get_ai_response(question_text, context_posts_text=context_for_ai)

    # 6. Fallback if somehow still empty
    if not response_message:
        response_message = "I'm having a bit of trouble thinking right now. Please try asking again."

    return QueryResponse(answer=response_message)

# To run this app:
# 1. Create a .env file in this directory with your SUPABASE_URL and SUPABASE_KEY.
#    Example .env file content:
#    SUPABASE_URL=your_supabase_url_here
#    SUPABASE_KEY=your_supabase_anon_key_here
#    OPENAI_API_KEY=your_openai_api_key_here
# 2. Ensure your tables are set up correctly:
#    - posts: with 'id' and 'text' columns
#    - comments: with 'post_id' and 'body' columns  
#    - attachments: with 'post_id' and 'extracted_text' columns
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run the app with: uvicorn chatbot:app --reload
>>>>>>> df68b0ae97619e32bdf8fecc171828f5721027a3
