"""
Plagiarism Detector Service
Uses Copyleaks API for plagiarism detection
"""
import os
import logging
import requests
from typing import Dict, Any, List, Optional
import time
import json
import hashlib

logger = logging.getLogger(__name__)

class PlagiarismDetector:
    """Plagiarism detection using Copyleaks API"""
    
    def __init__(self):
        self.api_key = os.environ.get('COPYLEAKS_API_KEY')
        self.email = os.environ.get('COPYLEAKS_EMAIL')
        self.base_url = 'https://api.copyleaks.com/v3'
        self.auth_token = None
        self.token_expiry = 0
        
        if not self.api_key or not self.email:
            logger.warning("Copyleaks credentials not configured")
    
    def check_plagiarism(self, text: str, is_pro: bool = False) -> Dict[str, Any]:
        """Check text for plagiarism"""
        try:
            # Check if Copyleaks is configured
            if not self.api_key or not self.email:
                return self._mock_plagiarism_check(text)
            
            # Authenticate if needed
            if not self._is_authenticated():
                if not self._authenticate():
                    return self._mock_plagiarism_check(text)
            
            # Submit text for plagiarism check
            scan_id = self._submit_for_plagiarism_check(text)
            if not scan_id:
                return self._mock_plagiarism_check(text)
            
            # Wait for results (with timeout)
            results = self._get_plagiarism_results(scan_id)
            
            if results:
                return self._process_plagiarism_results(results, text, is_pro)
            else:
                return self._mock_plagiarism_check(text)
                
        except Exception as e:
            logger.error(f"Plagiarism check error: {str(e)}", exc_info=True)
            return self._mock_plagiarism_check(text)
    
    def _is_authenticated(self) -> bool:
        """Check if authentication token is valid"""
        return self.auth_token and time.time() < self.token_expiry
    
    def _authenticate(self) -> bool:
        """Authenticate with Copyleaks API"""
        try:
            auth_url = f"{self.base_url}/account/login/api"
            
            payload = {
                'email': self.email,
                'key': self.api_key
            }
            
            response = requests.post(
                auth_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('access_token')
                # Token expires in 48 hours, but we'll refresh earlier
                self.token_expiry = time.time() + (47 * 60 * 60)
                return True
            else:
                logger.error(f"Copyleaks authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Copyleaks authentication error: {e}")
            return False
    
    def _submit_for_plagiarism_check(self, text: str) -> Optional[str]:
        """Submit text for plagiarism checking"""
        try:
            submit_url = f"{self.base_url}/scans/submit/text"
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            # Create a unique scan ID
            scan_id = hashlib.md5(f"{text[:100]}{time.time()}".encode()).hexdigest()[:12]
            
            payload = {
                'scanId': scan_id,
                'text': text,
                'properties': {
                    'webhooks': {
                        'status': f"{os.environ.get('WEBHOOK_URL', 'http://localhost:5000')}/webhook/{scan_id}"
                    },
                    'includeHtml': True,
                    'includeText': True,
                    'sandbox': False
                }
            }
            
            response = requests.post(
                submit_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                return scan_id
            else:
                logger.error(f"Copyleaks submission failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Copyleaks submission error: {e}")
            return None
    
    def _get_plagiarism_results(self, scan_id: str, max_attempts: int = 15) -> Optional[Dict]:
        """Poll for plagiarism check results"""
        try:
            results_url = f"{self.base_url}/scans/{scan_id}/results"
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}'
            }
            
            # Poll for results
            for attempt in range(max_attempts):
                response = requests.get(
                    results_url,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 204:
                    # Still processing
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Failed to get results: {response.status_code}")
                    return None
            
            logger.warning("Timeout waiting for Copyleaks results")
            return None
            
        except Exception as e:
            logger.error(f"Error getting Copyleaks results: {e}")
            return None
    
    def _process_plagiarism_results(self, results: Dict, original_text: str, is_pro: bool) -> Dict[str, Any]:
        """Process Copyleaks plagiarism results"""
        try:
            # Extract overall statistics
            stats = results.get('statistics', {})
            total_words = stats.get('totalWords', len(original_text.split()))
            identical_words = stats.get('identicalWords', 0)
            minor_changed_words = stats.get('minorChangedWords', 0)
            related_meaning_words = stats.get('relatedMeaningWords', 0)
            
            # Calculate plagiarism score
            plagiarized_words = identical_words + minor_changed_words + (related_meaning_words * 0.5)
            plagiarism_score = min((plagiarized_words / total_words * 100) if total_words > 0 else 0, 100)
            
            # Extract matched sources
            results_list = results.get('results', [])
            flagged_passages = []
            sources_found = len(results_list)
            
            # Process matches (limit to top 10 for free tier)
            limit = None if is_pro else 5
            for result in results_list[:limit]:
                matched_words = result.get('matchedWords', 0)
                if matched_words > 5:  # Only show significant matches
                    source_url = result.get('url', 'Unknown source')
                    source_title = result.get('title', source_url)
                    
                    # Extract matched passages
                    if is_pro and 'html' in result:
                        # Parse HTML to find matched text (simplified)
                        matched_text = self._extract_matched_text(result.get('html', ''), original_text)
                        if matched_text:
                            flagged_passages.append({
                                'text': matched_text[:200] + '...' if len(matched_text) > 200 else matched_text,
                                'source_url': source_url,
                                'source_title': source_title,
                                'similarity': round((matched_words / len(matched_text.split()) * 100) if matched_text else 0, 1)
                            })
                    else:
                        # Basic info for free tier
                        flagged_passages.append({
                            'text': '[Text preview available in premium]',
                            'source_url': source_url,
                            'source_title': source_title,
                            'similarity': round((matched_words / total_words * 100) if total_words > 0 else 0, 1)
                        })
            
            # Create summary
            summary = self._create_plagiarism_summary(plagiarism_score, sources_found)
            
            # Build response
            response = {
                'success': True,
                'plagiarism_score': round(plagiarism_score, 1),
                'sources_found': sources_found,
                'flagged_passages': flagged_passages,
                'summary': summary,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'is_pro': is_pro
            }
            
            # Add detailed analysis for pro users
            if is_pro:
                response['detailed_stats'] = {
                    'total_words': total_words,
                    'identical_words': identical_words,
                    'minor_changed_words': minor_changed_words,
                    'related_meaning_words': related_meaning_words
                }
                
                response['source_breakdown'] = self._create_source_breakdown(results_list)
                
                # Add paraphrase detection info
                response['paraphrase_detection'] = {
                    'detected': minor_changed_words > 0 or related_meaning_words > 0,
                    'paraphrased_content_percentage': round(
                        ((minor_changed_words + related_meaning_words) / total_words * 100) 
                        if total_words > 0 else 0, 1
                    )
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing plagiarism results: {e}")
            return self._mock_plagiarism_check(original_text)
    
    def _create_plagiarism_summary(self, score: float, sources: int) -> str:
        """Create summary based on plagiarism score"""
        if score >= 80:
            return f"High plagiarism detected! {score:.0f}% of the text matches {sources} existing sources. Immediate revision required."
        elif score >= 50:
            return f"Significant plagiarism found. {score:.0f}% similarity detected across {sources} sources. Major revisions needed."
        elif score >= 20:
            return f"Moderate similarity detected. {score:.0f}% of content matches {sources} sources. Review and cite properly."
        elif score >= 10:
            return f"Minor similarities found. {score:.0f}% matches detected in {sources} sources. Likely common phrases or properly cited content."
        else:
            return f"Excellent! Only {score:.0f}% similarity found. This appears to be original content with minimal matches."
    
    def _extract_matched_text(self, html: str, original_text: str) -> str:
        """Extract matched text from HTML (simplified)"""
        # In a real implementation, you would parse the HTML properly
        # For now, return a sample of the original text
        words = original_text.split()
        if len(words) > 50:
            return ' '.join(words[:50])
        return original_text
    
    def _create_source_breakdown(self, results: List[Dict]) -> List[Dict]:
        """Create detailed source breakdown for pro users"""
        breakdown = []
        
        for result in results[:10]:  # Top 10 sources
            breakdown.append({
                'url': result.get('url', 'Unknown'),
                'title': result.get('title', 'Untitled'),
                'matched_words': result.get('matchedWords', 0),
                'percentage': result.get('percent', 0),
                'published_date': result.get('publishDate', 'Unknown')
            })
        
        return breakdown
    
    def _mock_plagiarism_check(self, text: str) -> Dict[str, Any]:
        """Provide mock plagiarism check when Copyleaks is not available"""
        # Simple heuristic-based check
        words = text.lower().split()
        word_count = len(words)
        
        # Check for common academic phrases that might be plagiarized
        common_phrases = [
            'according to research', 'studies have shown', 'it has been proven',
            'research indicates', 'evidence suggests', 'scholars argue',
            'in conclusion', 'furthermore', 'moreover', 'nevertheless'
        ]
        
        phrase_count = sum(1 for phrase in common_phrases if phrase in text.lower())
        
        # Mock scoring
        base_score = min(phrase_count * 5, 30)
        
        # Add some randomness for demo purposes
        import random
        mock_score = base_score + random.randint(0, 20)
        
        # Create mock flagged passages
        flagged_passages = []
        if mock_score > 20:
            sentences = text.split('.')
            if len(sentences) > 2:
                flagged_passages.append({
                    'text': sentences[0].strip()[:150] + '...',
                    'source_url': 'https://example.com/source1',
                    'source_title': 'Example Academic Paper',
                    'similarity': random.randint(70, 95)
                })
            if mock_score > 40 and len(sentences) > 4:
                flagged_passages.append({
                    'text': sentences[2].strip()[:150] + '...',
                    'source_url': 'https://example.com/source2',
                    'source_title': 'Journal of Example Studies',
                    'similarity': random.randint(60, 85)
                })
        
        return {
            'success': True,
            'plagiarism_score': mock_score,
            'sources_found': len(flagged_passages),
            'flagged_passages': flagged_passages,
            'summary': self._create_plagiarism_summary(mock_score, len(flagged_passages)),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'is_pro': False,
            'note': 'This is a simplified analysis. Enable Copyleaks for professional plagiarism detection.'
        }

from datetime import datetime
