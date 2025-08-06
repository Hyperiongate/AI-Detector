"""
Text AI Analyzer Service
Analyzes text for AI-generated content patterns
"""
import os
import logging
import requests
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class TextAIAnalyzer:
    """Analyzes text content for AI generation patterns"""
    
    def __init__(self):
        self.scraper_api_key = os.environ.get('SCRAPERAPI_KEY')
        self.scrapingbee_api_key = os.environ.get('SCRAPINGBEE_API_KEY')
        
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """Extract text content from URL"""
        try:
            # Parse domain
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Try direct request first
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                # Try with scraping service
                if self.scraper_api_key:
                    response = self._fetch_with_scraperapi(url)
                elif self.scrapingbee_api_key:
                    response = self._fetch_with_scrapingbee(url)
            
            if response and response.status_code == 200:
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text
                text = self._extract_text_from_html(soup)
                
                # Extract metadata
                title = self._extract_title(soup)
                author = self._extract_author(soup)
                publish_date = self._extract_publish_date(soup)
                
                return {
                    'success': True,
                    'url': url,
                    'domain': domain,
                    'title': title,
                    'author': author,
                    'publish_date': publish_date,
                    'text': text,
                    'word_count': len(text.split()),
                    'content_type': response.headers.get('content-type', 'text/html')
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to fetch URL: {response.status_code if response else "No response"}'
                }
                
        except Exception as e:
            logger.error(f"URL extraction error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'Failed to extract content: {str(e)}'
            }
    
    def _fetch_with_scraperapi(self, url: str) -> Optional[requests.Response]:
        """Fetch URL using ScraperAPI"""
        try:
            api_url = 'http://api.scraperapi.com'
            params = {
                'api_key': self.scraper_api_key,
                'url': url,
                'render': 'false'
            }
            return requests.get(api_url, params=params, timeout=30)
        except Exception as e:
            logger.error(f"ScraperAPI error: {e}")
            return None
    
    def _fetch_with_scrapingbee(self, url: str) -> Optional[requests.Response]:
        """Fetch URL using ScrapingBee"""
        try:
            api_url = 'https://app.scrapingbee.com/api/v1'
            params = {
                'api_key': self.scrapingbee_api_key,
                'url': url,
                'render_js': 'false'
            }
            return requests.get(api_url, params=params, timeout=30)
        except Exception as e:
            logger.error(f"ScrapingBee error: {e}")
            return None
    
    def _extract_text_from_html(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        main_content = None
        
        # Common content selectors
        content_selectors = [
            'article', 'main', '[role="main"]', '.article-body',
            '.post-content', '.entry-content', '.content',
            '#content', '.story-body', '.article-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.body if soup.body else soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        # Try meta og:title first
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content']
        
        # Try regular title tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return 'Untitled'
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author information"""
        # Try meta author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            return author_meta['content']
        
        # Try schema.org
        author_schema = soup.find('span', itemprop='author')
        if author_schema:
            return author_schema.get_text(strip=True)
        
        # Try common author classes
        author_classes = ['.author', '.by-author', '.article-author', '.post-author']
        for cls in author_classes:
            author_elem = soup.select_one(cls)
            if author_elem:
                return author_elem.get_text(strip=True)
        
        return 'Unknown'
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publish date"""
        # Try meta article:published_time
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta and date_meta.get('content'):
            return date_meta['content']
        
        # Try datePublished schema
        date_schema = soup.find('time', itemprop='datePublished')
        if date_schema and date_schema.get('datetime'):
            return date_schema['datetime']
        
        # Try time tag
        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            return time_tag['datetime']
        
        return None
    
    def analyze_writing_style(self, text: str) -> Dict[str, Any]:
        """Analyze writing style characteristics"""
        sentences = self._split_sentences(text)
        words = text.split()
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary diversity
        unique_words = set(word.lower() for word in words if word.isalpha())
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # Sentence starters
        sentence_starters = [sent.split()[0].lower() for sent in sentences if sent.split()]
        repetitive_starters = self._calculate_repetition_rate(sentence_starters)
        
        # Common AI patterns
        ai_phrases = [
            'it is important to note',
            'it\'s worth noting',
            'in conclusion',
            'furthermore',
            'moreover',
            'however',
            'nevertheless',
            'in today\'s world',
            'in the modern era',
            'delve into',
            'tapestry',
            'landscape',
            'realm',
            'navigate',
            'leverage'
        ]
        
        ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text.lower())
        
        return {
            'avg_sentence_length': round(avg_sentence_length, 1),
            'vocabulary_diversity': round(vocabulary_diversity, 3),
            'repetitive_starters': round(repetitive_starters, 2),
            'ai_phrase_count': ai_phrase_count,
            'sentence_count': len(sentences),
            'word_count': len(words)
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_repetition_rate(self, items: List[str]) -> float:
        """Calculate repetition rate in a list"""
        if not items:
            return 0
        
        unique_items = set(items)
        return 1 - (len(unique_items) / len(items))
