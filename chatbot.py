import os
import base64
from supabase import create_client, Client
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Union
import uuid
import shutil
import tempfile
from PIL import Image
import pytesseract

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mortgage Information Assistant")

# Create temporary directory for image uploads
TEMP_DIR = tempfile.gettempdir()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Make sure index.html is in the same directory as chatbot.py.</p>", status_code=404)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return HTMLResponse(content=f"<h1>Internal Server Error</h1><p>Error serving HTML: {e}</p>", status_code=500)

# Initialize Supabase client from .env variables
supabase_url = os.getenv("SUPABASE_URL")
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

# Define request and response models
class QueryRequest(BaseModel):
    question: str
    image_data: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str

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
        print("poopy")
        logger.error(f"Error in vector search: {e}")
        return []

# Enhanced get_ai_response function for RAG with vision capabilities
async def get_ai_response(user_question: str, context: List[RetrievedContext], image_data: Optional[str] = None) -> str:
    """Generate AI response with retrieved context and optional image using RAG"""
    if not ai_client:
        return "I am currently unable to connect to the AI model to generate a response. Please check your OPENAI_API_KEY in the .env file."

    system_prompt = (
        "You are an AI Mortgage Information Assistant. Provide concise, helpful answers about mortgage topics.\n\n"
        
        "You have access to:\n"
        "1. General mortgage knowledge\n"
        "2. Reddit posts from r/firsttimehomebuyer\n"
        "3. Community comments and discussions\n"
        "4. Images shared by the user\n\n"
        "5. data from the database\n\n"
        
        "Response guidelines:\n"
        "- Keep answers brief and to the point (2-3 paragraphs max)\n"
        "- Only include the most relevant information\n"
        "- Mention sources briefly: 'Based on Reddit posts...' or 'From the image shared...'\n"
        "- Focus on key numbers or insights, not exhaustive details\n"
        "- If the user wants more detail, they can ask follow-up questions\n"
        "- Remember this is community-shared content, not official financial advice\n\n"
        
        "Be conversational and helpful, but keep it concise!"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]

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
    else:
        messages.append({"role": "user", "content": user_question})

    try:
        logger.info(f"Sending request to AI model. User question: '{user_question}', Context provided: {bool(context)}, Image provided: {bool(image_data)}")
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
async def ask_question(request: QueryRequest):
    question_text = request.question
    image_data = request.image_data
    
    if not question_text or question_text.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check for database statistics requests
    if any(phrase in question_text.lower() for phrase in ["database stats", "how many posts", "how many comments", "how many attachments"]):
        stats = await get_database_stats()
        stats_response = (
            f"Current database statistics:\n"
            f"• Posts: {stats['posts']}\n"
            f"• Comments: {stats['comments']}\n"
            f"• Attachments: {stats['attachments']}"
        )
        return QueryResponse(answer=stats_response)
    
    # Generate embedding for the question
    query_embedding = await generate_embedding(question_text)
    
    if not query_embedding:
        # Fallback to general response if embedding generation fails
        fallback_response = "I'm having trouble processing your question semantically. I can still provide general mortgage information, but personalized results are currently unavailable."
        return QueryResponse(answer=fallback_response)
    
    # Perform vector search using embeddings
    search_results = await vector_search(query_embedding)
    
    # Generate AI response with retrieved context and image if available
    response_message = await get_ai_response(question_text, search_results, image_data)
    
    return QueryResponse(answer=response_message)

# Entry point for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)

# Note: For the vector similarity search to work, you need to create the following SQL functions in your Supabase database:
# 1. match_posts_embeddings(query_embedding vector, match_threshold float, match_count int)
# 2. match_comments_embeddings(query_embedding vector, match_threshold float, match_count int)
# 3. match_attachments_embeddings(query_embedding vector, match_threshold float, match_count int)