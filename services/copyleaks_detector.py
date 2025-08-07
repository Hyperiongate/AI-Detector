"""
Copyleaks AI Detector Service
Integrates with Copyleaks API for professional AI detection
"""
import os
import logging
import requests
from typing import Dict, Any, Optional, List
import time
import json

logger = logging.getLogger(__name__)

class CopyleaksDetector:
    """Copyleaks AI detection integration"""
    
    def __init__(self):
        self.api_key = os.environ.get('COPYLEAKS_API_KEY')
        self.email = os.environ.get('COPYLEAKS_EMAIL')
        self.base_url = 'https://api.copyleaks.com/v3'
        self.auth_token = None
        self.token_expiry = 0
        
        if not self.api_key or not self.email:
            logger.warning("Copyleaks credentials not configured")
    
    def detect_ai(self, text: str) -> Dict[str, Any]:
        """Detect AI-generated content using Copyleaks"""
        try:
            # Check if Copyleaks is configured
            if not self.api_key or not self.email:
                return self._mock_detection(text)
            
            # Authenticate if needed
            if not self._is_authenticated():
                if not self._authenticate():
                    return self._mock_detection(text)
            
            # Submit text for AI detection
            scan_id = self._submit_for_detection(text)
            if not scan_id:
                return self._mock_detection(text)
            
            # Wait for results (with timeout)
            results = self._get_detection_results(scan_id)
            
            if results:
                return self._process_results(results, text)
            else:
                return self._mock_detection(text)
                
        except Exception as e:
            logger.error(f"Copyleaks detection error: {str(e)}", exc_info=True)
            return self._mock_detection(text)
    
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
    
    def _submit_for_detection(self, text: str) -> Optional[str]:
        """Submit text for AI detection"""
        try:
            submit_url = f"{self.base_url}/ai-content-detection/submit/text"
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'text': text,
                'sandbox': False  # Use production mode
            }
            
            response = requests.post(
                submit_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                return data.get('scanId')
            else:
                logger.error(f"Copyleaks submission failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Copyleaks submission error: {e}")
            return None
    
    def _get_detection_results(self, scan_id: str, max_attempts: int = 10) -> Optional[Dict]:
        """Poll for detection results"""
        try:
            results_url = f"{self.base_url}/ai-content-detection/{scan_id}/result"
            
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
    
    def _process_results(self, results: Dict, original_text: str) -> Dict[str, Any]:
        """Process Copyleaks AI detection results"""
        try:
            # Extract AI probability from results
            # Copyleaks returns a score between 0-100 for AI probability
            ai_probability = results.get('ai_score', 50)
            
            # Get summary information
            summary_data = results.get('summary', {})
            
            # Extract section-level results if available
            sections = results.get('sections', [])
            detected_sections = []
            
            for section in sections:
                if section.get('ai_score', 0) > 70:  # High AI probability sections
                    text_snippet = section.get('text', '')[:100] + '...'
                    detected_sections.append({
                        'text': text_snippet,
                        'ai_score': section.get('ai_score', 0),
                        'start': section.get('start', 0),
                        'end': section.get('end', 0)
                    })
            
            # Calculate confidence based on various factors
            confidence = 'high' if len(sections) > 5 else 'medium'
            
            return {
                'ai_probability': ai_probability,
                'summary': self._create_summary(ai_probability),
                'detected_sections': [s['text'] for s in detected_sections[:3]],  # Top 3 sections
                'confidence': confidence,
                'total_words_analyzed': len(original_text.split()),
                'ai_words_detected': summary_data.get('ai_words', 0),
                'human_words_detected': summary_data.get('human_words', 0),
                'analysis_version': results.get('version', 'latest'),
                'section_analysis': detected_sections if detected_sections else None
            }
            
        except Exception as e:
            logger.error(f"Error processing Copyleaks results: {e}")
            return self._mock_detection("")
    
    def _create_summary(self, ai_probability: float) -> str:
        """Create summary based on AI probability"""
        if ai_probability >= 90:
            return "This content is almost certainly AI-generated according to Copyleaks analysis."
        elif ai_probability >= 70:
            return "High probability of AI generation detected by Copyleaks professional detection."
        elif ai_probability >= 50:
            return "Moderate AI involvement detected. Content may be AI-assisted or partially generated."
        elif ai_probability >= 30:
            return "Some AI indicators present, but content appears predominantly human-written."
        else:
            return "Content appears to be human-written with minimal AI involvement."
    
    def _mock_detection(self, text: str) -> Dict[str, Any]:
        """Provide mock detection when Copyleaks is not available"""
        # Simple heuristic-based detection as fallback
        ai_indicators = [
            'in conclusion', 'furthermore', 'moreover', 'it is important to note',
            'delve into', 'it\'s worth noting', 'in today\'s', 'in the modern',
            'tapestry', 'landscape', 'realm', 'navigate', 'leverage',
            'it is crucial', 'it is essential', 'comprehensive', 'multifaceted'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in ai_indicators if indicator in text_lower)
        
        # Check for perfect grammar patterns
        sentences = text.split('.')
        perfect_sentences = 0
        for sentence in sentences:
            if sentence.strip() and len(sentence.strip()) > 20:
                # Check if sentence starts with capital and has no obvious errors
                if sentence.strip()[0].isupper():
                    perfect_sentences += 1
        
        # Calculate mock probability
        words = text.split()
        if len(words) > 0:
            indicator_rate = indicator_count / (len(words) / 100)
            grammar_score = (perfect_sentences / len(sentences)) * 30 if sentences else 0
            ai_probability = min(95, indicator_rate * 15 + grammar_score + 20)
        else:
            ai_probability = 50
        
        # Create mock sections for demonstration
        detected_sections = []
        if ai_probability > 60 and len(sentences) > 3:
            detected_sections = [
                sentences[0].strip()[:100] + '...' if sentences[0] else '',
                sentences[len(sentences)//2].strip()[:100] + '...' if len(sentences) > 2 else ''
            ]
        
        return {
            'ai_probability': ai_probability,
            'summary': 'Analysis performed using pattern matching (Copyleaks unavailable)',
            'detected_sections': [s for s in detected_sections if s],
            'confidence': 'low',
            'note': 'This is a simplified analysis. Enable Copyleaks for professional AI detection.',
            'total_words_analyzed': len(words),
            'ai_words_detected': int(len(words) * (ai_probability / 100)),
            'human_words_detected': int(len(words) * ((100 - ai_probability) / 100))
        }
