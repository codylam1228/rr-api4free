"""
Combined server for Free OpenRouter API Key Pool Manager
All logic from config.py, key_manager.py, key_pool.py, models.py, main.py, and routes.py is merged here.
"""

import os
import threading
import logging
import time
import requests
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from functools import wraps

# --- config.py ---

class Config:
    """Application configuration"""
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8964
    LOG_LEVEL: str = "INFO"
    KEYS_FILE = "./openrouter/keys.ini"

    @classmethod
    def get_api_keys(cls) -> List[str]:
        # Only load from ./openrouter/keys.ini
        if not os.path.exists(cls.KEYS_FILE):
            print(f"Error: {cls.KEYS_FILE} not found. Please create it and add your API keys (one per line).")
            return []
        keys = []
        with open(cls.KEYS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("sk-or-v1-"):
                    keys.append(line)
                else:
                    print(f"Warning: Invalid API key format in {cls.KEYS_FILE}: {line}")
        if not keys:
            print(f"No valid API keys found in {cls.KEYS_FILE}.")
        else:
            print(f"Loaded {len(keys)} API keys from {cls.KEYS_FILE}")
        return keys

    @classmethod
    def get_api_key_count(cls) -> int:
        return len(cls.get_api_keys())

config = Config()

# --- models.py ---
class APIKey(BaseModel):
    id: int = Field(..., description="Unique identifier for the key")
    value: str = Field(..., description="The actual API key value")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the key was added")
    last_used: Optional[datetime] = Field(None, description="Last time the key was used")
    usage_count: int = Field(default=0, description="Number of times the key has been used")

class KeyPoolStats(BaseModel):
    total_keys: int = Field(..., description="Total number of keys in pool")
    active_keys: int = Field(..., description="Number of active keys")
    total_usage: int = Field(..., description="Total usage across all keys")

class NextKeyResponse(BaseModel):
    key: str = Field(..., description="The API key")
    index: int = Field(..., description="Index of the key in pool")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    keys_count: int = Field(..., description="Number of available keys")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# --- key_manager.py ---
logger = logging.getLogger("key_manager")

class KeyManager:
    def __init__(self):
        self._keys: List[str] = []
        self._load_keys()

    def _load_keys(self):
        try:
            self._keys = config.get_api_keys()
            logger.info(f"Loaded {len(self._keys)} API keys")
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            self._keys = []

    def get_keys(self) -> List[int]:
        return [i + 1 for i in range(len(self._keys))]

    def get_key_count(self) -> int:
        return len(self._keys)

    def get_key_by_index(self, index: int) -> Optional[str]:
        if 0 <= index < len(self._keys):
            logger.debug(f"Retrieved key at index {index}")
            return self._keys[index]
        logger.warning(f"Invalid key index: {index}")
        return None

    def reload_keys(self):
        logger.info("Reloading API keys from configuration")
        self._load_keys()


key_manager = KeyManager()

# --- key_pool.py ---
logger_pool = logging.getLogger("key_pool")

class KeyPool:
    def __init__(self):
        self._keys: List[APIKey] = []
        self._current_index: int = 0
        self._lock = threading.Lock()
        self._load_keys()

    def _load_keys(self):
        try:
            raw_keys = key_manager.get_keys()
            self._keys = []
            for i, key_id in enumerate(raw_keys):
                key_value = key_manager.get_key_by_index(i)
                if key_value:
                    api_key = APIKey(
                        id=key_id,
                        value=key_value,
                        created_at=datetime.utcnow()
                    )
                    self._keys.append(api_key)
            logger_pool.info(f"Loaded {len(self._keys)} keys into pool")
        except Exception as e:
            logger_pool.error(f"Failed to load keys into pool: {e}")
            self._keys = []

    def get_next_key(self) -> Optional[APIKey]:
        with self._lock:
            if not self._keys:
                logger_pool.warning("No keys available in pool")
                return None
            current_key = self._keys[self._current_index]
            current_key.last_used = datetime.utcnow()
            current_key.usage_count += 1
            self._current_index = (self._current_index + 1) % len(self._keys)
            logger_pool.debug(f"Selected key {current_key.id}, usage count: {current_key.usage_count}")
            return current_key


    def get_keys_info(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "id": key.id,
                    "key": key.value,
                    "usage_count": key.usage_count
                }
                for key in self._keys
            ]

    def get_pool_stats(self) -> dict:
        with self._lock:
            total_usage = sum(key.usage_count for key in self._keys)
            return {
                "total_keys": len(self._keys),
                "active_keys": len(self._keys),
                "total_usage": total_usage,
                "current_index": self._current_index
            }

    def reload_keys(self):
        with self._lock:
            logger_pool.info("Reloading keys into pool")
            self._load_keys()
            self._current_index = 0

key_pool = KeyPool()

# --- FreeAPI Client Implementation ---
class FreeAPIClient:
    """
    Production-ready FreeAPI client with intelligent caching and error handling.
    
    Features:
    - Connection pooling with requests.Session
    - Intelligent caching (5-minute default)
    - Retry logic with exponential backoff
    - Rate limiting
    - Circuit breaker pattern
    - Health monitoring
    """
    
    def __init__(self, base_url: str = "http://localhost:8964", cache_duration: int = 300):
        self.base_url = base_url
        self.session = requests.Session()
        self.cache_duration = cache_duration
        
        # Caching
        self.cached_key = None
        self.key_expiry = 0
        self.key_usage_count = 0
        self.max_usage_per_key = 50
        
        # Circuit breaker
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        
        # Rate limiting
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        logger.info(f"Initialized FreeAPI client: {self.base_url}")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.failure_count >= self.circuit_breaker_threshold:
            if time.time() - self.last_failure_time < self.circuit_breaker_timeout:
                return True
            else:
                # Reset circuit breaker
                self.failure_count = 0
        return False
    
    def _record_success(self):
        """Record successful request"""
        self.failure_count = 0
    
    def _record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
    
    def get_api_key(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get API key with intelligent caching and error handling
        
        Args:
            force_refresh: Force refresh of cached key
            
        Returns:
            API key string or None if failed
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is open, skipping request")
            return None
        
        # Rate limiting
        self._rate_limit()
        
        current_time = time.time()
        
        # Return cached key if still valid
        if (not force_refresh and 
            self.cached_key and 
            current_time < self.key_expiry and 
            self.key_usage_count < self.max_usage_per_key):
            self.key_usage_count += 1
            return self.cached_key
        
        # Fetch new key with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v1/keys/next",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.cached_key = data["key"]
                    self.key_expiry = current_time + self.cache_duration
                    self.key_usage_count = 1
                    self._record_success()
                    
                    logger.info(f"Got fresh API key (attempt {attempt + 1})")
                    return self.cached_key
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        self._record_failure()
        logger.error("Failed to get API key after all retries")
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/health",
                timeout=5
            )
            return response.json() if response.status_code == 200 else {"status": "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/stats",
                timeout=5
            )
            return response.json() if response.status_code == 200 else {"error": "Stats failed"}
        except Exception as e:
            return {"error": str(e)}
    
    def with_api_key(self, func: Callable) -> Callable:
        """
        Decorator to automatically inject API key into function
        
        Usage:
            @client.with_api_key
            def my_function(api_key: str, data: dict):
                # Your function here
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            api_key = self.get_api_key()
            if not api_key:
                return {"error": "No API key available"}
            
            kwargs["api_key"] = api_key
            return func(*args, **kwargs)
        
        return wrapper
    
    def make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with automatic API key injection
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data or error dict
        """
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No API key available"}
        
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {api_key}'
        kwargs['headers'] = headers
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def ask_question(self, question: str, model: str = "deepseek/deepseek-chat:free") -> Dict[str, Any]:
        """
        Ask a question using OpenRouter API with free models
        
        Args:
            question: The question to ask
            model: Model to use (default: deepseek/deepseek-chat:free)
            
        Returns:
            Response with answer or error
        """
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No API key available"}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8964",
            "X-Title": "FreeAPI Client"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "question": question,
                    "answer": result['choices'][0]['message']['content'],
                    "model": model,
                    "key_used": api_key[:20] + "...",
                    "success": True
                }
            else:
                return {
                    "question": question,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "success": False
                }
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "success": False
            }

# Global FreeAPI client instance
freeapi_client = FreeAPIClient()

# --- routes.py ---
router = APIRouter()

@router.get("/keys/next", response_model=NextKeyResponse)
async def get_next_key():
    try:
        api_key = key_pool.get_next_key()
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No API keys available in pool"
            )
        index = api_key.id
        return NextKeyResponse(key=api_key.value, index=index)
    except HTTPException:
        raise
    except Exception as e:
        logger_pool.error(f"Error getting next key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        pool_stats = key_pool.get_pool_stats()
        return HealthResponse(
            status="healthy",
            keys_count=pool_stats["total_keys"]
        )
    except Exception as e:
        logger_pool.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unhealthy"
        )

@router.get("/stats")
async def get_stats():
    try:
        stats = key_pool.get_pool_stats()
        keys_info = key_pool.get_keys_info()
        return {
            "pool_stats": stats,
            "keys_info": keys_info
        }
    except Exception as e:
        logger_pool.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# --- main.py ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger_main = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger_main.info("Starting Free OpenRouter API Key Pool Manager")
    logger_main.info(f"Loaded {config.get_api_key_count()} API keys")
    yield
    logger_main.info("Shutting down Free OpenRouter API Key Pool Manager")

app = FastAPI(
    title="Free OpenRouter API Key Pool Manager",
    description="""
    A service to manage and distribute free OpenRouter API keys using round-robin algorithm.

    ## Features
    - Plug-and-play API key setup with ./openrouter/keys.ini
    - Round-robin key distribution for even usage
    - Dynamic key pool management
    - RESTful API with OpenAPI documentation
    - Health monitoring and statistics

    ## Usage
    Add your OpenRouter API keys to a file named ./openrouter/keys.ini in the project root:
    - Each line should be a key (sk-or-v1-...)
    - Blank lines and lines starting with # are ignored
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Free OpenRouter API Key Pool Manager",
        "description": "API key management service"
    },
    license_info={
        "name": "MIT"
    },
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["API Keys"])

@app.get("/")
async def root():
    return {
        "message": "Free OpenRouter API Key Pool Manager",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "API Key Pool Manager"}

if __name__ == "__main__":
    import uvicorn
    logger_main.info(f"Starting server on {config.SERVICE_HOST}:{config.SERVICE_PORT}")
    uvicorn.run(
        app,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        log_level=config.LOG_LEVEL.lower()
    )

# --- Usage Examples ---
"""
FreeAPI Client Usage Examples
============================

The FreeAPI client is now built into server.py and ready to use!

1. BASIC USAGE:
   ```python
   from server import freeapi_client
   
   # Get an API key
   api_key = freeapi_client.get_api_key()
   if api_key:
       print(f"Got key: {api_key[:20]}...")
   ```

2. USING DECORATOR:
   ```python
   from server import freeapi_client
   
   @freeapi_client.with_api_key
   def call_external_api(api_key: str, data: dict):
       headers = {"Authorization": f"Bearer {api_key}"}
       response = requests.get("https://api.example.com/data", headers=headers)
       return response.json()
   
   result = call_external_api(data={"test": "data"})
   ```

3. DIRECT REQUEST:
   ```python
   from server import freeapi_client
   
   result = freeapi_client.make_request("GET", "https://httpbin.org/headers")
   print(result)
   ```

4. HEALTH MONITORING:
   ```python
   from server import freeapi_client
   
   health = freeapi_client.health_check()
   stats = freeapi_client.get_stats()
   print(f"Health: {health}")
   print(f"Stats: {stats}")
   ```

5. OPENROUTER INTEGRATION:
   ```python
   from server import freeapi_client
   
   def call_openrouter(prompt: str):
       api_key = freeapi_client.get_api_key()
       if not api_key:
           return {"error": "No API key available"}
       
       headers = {
           "Authorization": f"Bearer {api_key}",
           "Content-Type": "application/json"
       }
       
       data = {
           "model": "openai/gpt-3.5-turbo",
           "messages": [{"role": "user", "content": prompt}]
       }
       
       response = requests.post(
           "https://openrouter.ai/api/v1/chat/completions",
           headers=headers,
           json=data
       )
       return response.json()
   
   result = call_openrouter("Hello, world!")
   ```

6. CUSTOM CONFIGURATION:
   ```python
   from server import FreeAPIClient
   
   # Create custom client with different settings
   custom_client = FreeAPIClient(
       base_url="http://localhost:8964",
       cache_duration=600  # 10 minutes cache
   )
   
   api_key = custom_client.get_api_key()
   ```

FEATURES INCLUDED:
- ✅ Connection pooling with requests.Session
- ✅ Intelligent caching (5-minute default)
- ✅ Retry logic with exponential backoff
- ✅ Rate limiting (100ms between requests)
- ✅ Circuit breaker pattern
- ✅ Health monitoring
- ✅ Automatic key rotation
- ✅ Error handling and logging

For more details, see the README.md file.
"""
