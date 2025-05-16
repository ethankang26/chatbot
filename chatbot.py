import os
from supabase import create_client, Client
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mortgage Information Assistant")

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

class QueryResponse(BaseModel):
    answer: str

class RetrievedContext(BaseModel):
    source: str
    content: str
    post_id: Optional[int] = None

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-3.5-turbo"
EMBEDDING_DIMENSION = 1536  # Dimension of text-embedding-3-small
SIMILARITY_THRESHOLD = 0.7
MAX_RESULTS_PER_TABLE = 3
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
            for post in posts_rpc.data:
                post_id = post.get('id')
                text = post.get('text', '')
                title = post.get('title', 'Reddit Post')
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
            for comment in comments_rpc.data:
                post_id = comment.get('post_id')
                body = comment.get('body', '')
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
            for attachment in attachments_rpc.data:
                post_id = attachment.get('post_id')
                text = attachment.get('extracted_text', '')
                results.append(RetrievedContext(
                    source=f"Document from Post {post_id}",
                    content=text,
                    post_id=post_id
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []

# Enhanced get_ai_response function for RAG
async def get_ai_response(user_question: str, context: List[RetrievedContext]) -> str:
    """Generate AI response with retrieved context using RAG"""
    if not ai_client:
        return "I am currently unable to connect to the AI model to generate a response. Please check your OPENAI_API_KEY in the .env file."

    system_prompt = (
        "You are an AI Mortgage Information Assistant. Provide concise, helpful answers about mortgage topics.\n\n"
        
        "You have access to:\n"
        "1. General mortgage knowledge\n"
        "2. Reddit posts from r/firsttimehomebuyer\n"
        "3. Community comments and discussions\n"
        "4. Extracted text from mortgage documents shared as images\n\n"
        
        "Response guidelines:\n"
        "- Keep answers brief and to the point (2-3 paragraphs max)\n"
        "- Only include the most relevant information\n"
        "- Mention sources briefly: 'Based on Reddit posts...' or 'From mortgage documents shared...'\n"
        "- Focus on key numbers or insights, not exhaustive details\n"
        "- If the user wants more detail, they can ask follow-up questions\n"
        "- Remember this is community-shared content, not official financial advice\n\n"
        
        "Be conversational and helpful, but keep it concise!"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
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
        messages.insert(1, {"role": "system", "content": context_message})

    try:
        logger.info(f"Sending request to AI model. User question: '{user_question}', Context provided: {bool(context)}")
        completion = ai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300
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

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    question_text = request.question
    
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
    
    # Generate AI response with retrieved context
    response_message = await get_ai_response(question_text, search_results)
    
    return QueryResponse(answer=response_message)

# Entry point for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)

# Note: For the vector similarity search to work, you need to create the following SQL functions in your Supabase database:
# 1. match_posts_embeddings(query_embedding vector, match_threshold float, match_count int)
# 2. match_comments_embeddings(query_embedding vector, match_threshold float, match_count int)
# 3. match_attachments_embeddings(query_embedding vector, match_threshold float, match_count int)