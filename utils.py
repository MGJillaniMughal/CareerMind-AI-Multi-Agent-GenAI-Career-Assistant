"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Utility Functions with Advanced Features and System Integration
"""

import os
import sys
import platform
import psutil
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import hashlib
import re
from pathlib import Path
import requests
from urllib.parse import urlparse, urljoin
import time

# External service imports
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import FireCrawlLoader
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EnhancedSerperClient:
    """
    Enhanced Google search client with advanced features and error handling.
    """
    
    def __init__(self, serper_api_key: str = None):
        """
        Initialize the enhanced Serper client.
        
        Args:
            serper_api_key: API key for Serper service
        """
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = 0
        self.request_count = 0
        self.daily_limit = 1000
        
        if not self.serper_api_key:
            logger.warning("Serper API key not found. Search functionality will be limited.")
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search",
        location: str = None,
        time_period: str = None,
        domain_filter: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced search with advanced filtering and error handling.
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-100)
            search_type: Type of search (search, news, images, places)
            location: Geographic location for search
            time_period: Time period filter (hour, day, week, month, year)
            domain_filter: Domain to restrict search to
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Validate inputs
            if not query or len(query.strip()) == 0:
                raise ValueError("Search query cannot be empty")
            
            num_results = max(1, min(num_results, 100))
            
            # Build search parameters
            search_params = {
                "q": query.strip(),
                "num": num_results,
                "type": search_type
            }
            
            if location:
                search_params["location"] = location
            if time_period:
                search_params["tbs"] = f"qdr:{time_period[0]}"  # h, d, w, m, y
            if domain_filter:
                search_params["q"] += f" site:{domain_filter}"
            
            # Perform search
            if self.serper_api_key:
                wrapper = GoogleSerperAPIWrapper(
                    serper_api_key=self.serper_api_key,
                    k=num_results
                )
                response = wrapper.results(query=search_params["q"])
                
                # Process and enhance response
                items = response.pop("organic", [])
                response["items"] = items
                response["query_metadata"] = {
                    "original_query": query,
                    "processed_query": search_params["q"],
                    "num_results_requested": num_results,
                    "num_results_returned": len(items),
                    "search_type": search_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.request_count += 1
                logger.info(f"Search completed: '{query}' returned {len(items)} results")
                
                return response
            else:
                # Fallback response when API key is not available
                return self._create_fallback_response(query, num_results)
                
        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}")
            return self._create_error_response(query, str(e))
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _create_fallback_response(self, query: str, num_results: int) -> Dict[str, Any]:
        """Create fallback response when API is unavailable"""
        return {
            "items": [],
            "query_metadata": {
                "original_query": query,
                "num_results_requested": num_results,
                "num_results_returned": 0,
                "timestamp": datetime.now().isoformat(),
                "status": "api_unavailable"
            },
            "searchParameters": {"q": query},
            "error": "Serper API key not configured"
        }
    
    def _create_error_response(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "items": [],
            "query_metadata": {
                "original_query": query,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error_message": error_msg
            },
            "error": error_msg
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "request_count": self.request_count,
            "daily_limit": self.daily_limit,
            "remaining_requests": max(0, self.daily_limit - self.request_count),
            "api_configured": bool(self.serper_api_key)
        }


class EnhancedFireCrawlClient:
    """
    Enhanced web scraping client with advanced content processing.
    """
    
    def __init__(self, firecrawl_api_key: str = None):
        """
        Initialize the enhanced FireCrawl client.
        
        Args:
            firecrawl_api_key: API key for FireCrawl service
        """
        self.firecrawl_api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        if not self.firecrawl_api_key:
            logger.warning("FireCrawl API key not found. Web scraping will use fallback methods.")
    
    def scrape(
        self,
        url: str,
        max_length: int = 10000,
        extract_metadata: bool = True,
        follow_links: bool = False,
        clean_content: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced web scraping with content processing and metadata extraction.
        
        Args:
            url: URL to scrape
            max_length: Maximum content length to return
            extract_metadata: Whether to extract page metadata
            follow_links: Whether to follow and scrape linked pages
            clean_content: Whether to clean and format content
            
        Returns:
            Dict containing scraped content and metadata
        """
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Check cache
            cache_key = self._generate_cache_key(url, max_length)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.info(f"Returning cached content for: {url}")
                    return cached_result
            
            # Scrape content
            if self.firecrawl_api_key:
                content_data = self._scrape_with_firecrawl(url, max_length)
            else:
                content_data = self._scrape_with_fallback(url, max_length)
            
            # Process content
            if clean_content:
                content_data["content"] = self._clean_content(content_data["content"])
            
            # Extract metadata if requested
            if extract_metadata:
                content_data["metadata"] = self._extract_metadata(content_data["content"], url)
            
            # Cache result
            content_data["cached_at"] = datetime.now().isoformat()
            self.cache[cache_key] = content_data
            
            logger.info(f"Successfully scraped: {url} ({len(content_data['content'])} chars)")
            return content_data
            
        except Exception as e:
            logger.error(f"Scraping error for {url}: {str(e)}")
            return {
                "content": "",
                "error": str(e),
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def _scrape_with_firecrawl(self, url: str, max_length: int) -> Dict[str, Any]:
        """Scrape using FireCrawl API"""
        try:
            docs = FireCrawlLoader(
                api_key=self.firecrawl_api_key,
                url=url,
                mode="scrape"
            ).lazy_load()
            
            content = ""
            for doc in docs:
                content += doc.page_content
                if len(content) >= max_length:
                    break
            
            return {
                "content": content[:max_length],
                "url": url,
                "method": "firecrawl",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"FireCrawl scraping failed: {str(e)}")
            raise
    
    def _scrape_with_fallback(self, url: str, max_length: int) -> Dict[str, Any]:
        """Fallback scraping method using requests and basic parsing"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            content = soup.get_text()
            
            return {
                "content": content[:max_length],
                "url": url,
                "method": "fallback",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Fallback scraping failed: {str(e)}")
            raise
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _generate_cache_key(self, url: str, max_length: int) -> str:
        """Generate cache key for URL and parameters"""
        key_string = f"{url}_{max_length}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        if "cached_at" not in cached_result:
            return False
        
        cached_time = datetime.fromisoformat(cached_result["cached_at"])
        return datetime.now() - cached_time < timedelta(seconds=self.cache_duration)
    
    def _clean_content(self, content: str) -> str:
        """Clean and format scraped content"""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might cause issues
        content = re.sub(r'[^\w\s\-.,;:!?()[\]{}"\'/\\@#$%^&*+=<>~`|]', '', content)
        
        # Normalize line breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _extract_metadata(self, content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        try:
            domain = urlparse(url).netloc
            
            # Basic content analysis
            words = content.split()
            sentences = content.split('.')
            
            return {
                "domain": domain,
                "word_count": len(words),
                "sentence_count": len(sentences),
                "content_length": len(content),
                "language": "en",  # Could be enhanced with language detection
                "extracted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return {}


# Legacy compatibility classes
class SerperClient(EnhancedSerperClient):
    """Backward compatibility wrapper"""
    def __init__(self, serper_api_key: str = None):
        super().__init__(serper_api_key)


class FireCrawlClient(EnhancedFireCrawlClient):
    """Backward compatibility wrapper"""
    def __init__(self, firecrawl_api_key: str = None):
        super().__init__(firecrawl_api_key)


# System Information and Utilities
def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for debugging and monitoring.
    
    Returns:
        Dict containing system information
    """
    try:
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent_used": psutil.virtual_memory().percent,
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "cpu": {
                "count": psutil.cpu_count(),
                "physical_count": psutil.cpu_count(logical=False),
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1)
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent_used": psutil.disk_usage('/').percent,
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
            },
            "environment": {
                "python_executable": sys.executable,
                "python_path": sys.path[:3],  # First 3 paths
                "working_directory": os.getcwd(),
                "user": os.environ.get("USER", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate availability and format of API keys.
    
    Returns:
        Dict mapping API service names to availability status
    """
    api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "groq": os.environ.get("GROQ_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "serper": os.environ.get("SERPER_API_KEY"),
        "firecrawl": os.environ.get("FIRECRAWL_API_KEY"),
        "langchain": os.environ.get("LANGCHAIN_API_KEY")
    }
    
    validation_results = {}
    
    for service, key in api_keys.items():
        if not key:
            validation_results[service] = False
        elif len(key) < 10:  # Basic length check
            validation_results[service] = False
        else:
            validation_results[service] = True
    
    # Special validation for specific services
    if validation_results.get("openai") and api_keys["openai"]:
        validation_results["openai"] = api_keys["openai"].startswith("sk-")
    
    logger.info(f"API key validation: {sum(validation_results.values())}/{len(validation_results)} keys available")
    return validation_results


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def clean_filename(filename: str) -> str:
    """
    Clean filename to remove invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores and dots
    filename = filename.strip('_.')
    
    # Ensure filename is not empty
    if not filename:
        filename = "untitled"
    
    return filename


def generate_unique_filename(base_name: str, extension: str, directory: str = "temp") -> str:
    """
    Generate a unique filename to avoid conflicts.
    
    Args:
        base_name: Base filename without extension
        extension: File extension (with or without dot)
        directory: Target directory
        
    Returns:
        Unique filename with path
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    base_name = clean_filename(base_name)
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    
    counter = 1
    while True:
        if counter == 1:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}_{counter}{extension}"
        
        full_path = directory / filename
        if not full_path.exists():
            return str(full_path)
        
        counter += 1
        if counter > 1000:  # Prevent infinite loop
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}{extension}"
            return str(directory / filename)


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {str(e)}")
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        default: Default JSON string if serialization fails
        
    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {str(e)}")
        return default


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity score.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating text similarity: {str(e)}")
        return 0.0


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    try:
        if not text:
            return []
        
        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their'
        }
        
        # Extract words and count frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, freq in sorted_keywords[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []


def create_performance_monitor():
    """
    Create a simple performance monitoring context manager.
    
    Returns:
        Context manager for performance monitoring
    """
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.duration = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
        
        def get_duration(self) -> float:
            return self.duration if self.duration else 0.0
        
        def get_formatted_duration(self) -> str:
            duration = self.get_duration()
            if duration < 1:
                return f"{duration*1000:.1f}ms"
            elif duration < 60:
                return f"{duration:.2f}s"
            else:
                return f"{duration/60:.1f}m"
    
    return PerformanceMonitor()


# Environment setup utilities
def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup enhanced logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    import logging.handlers
    
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def ensure_directories_exist(directories: List[str]) -> None:
    """
    Ensure required directories exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


# Default utility instances for backward compatibility
serper_client = EnhancedSerperClient()
firecrawl_client = EnhancedFireCrawlClient()