"""
AI Detector Service
Main orchestrator for AI detection analysis
"""
import os
import logging
from typing import Dict, Any, Optional, List
import requests
from datetime import datetime
import base64
from io import BytesIO

# Import analysis components
from .text_ai_analyzer import TextAIAnalyzer
from .image_ai_analyzer import ImageAIAnalyzer
from .pattern_analyzer import PatternAnalyzer
from .perplexity_analyzer import PerplexityAnalyzer
from .copyleaks_detector import CopyleaksDetector
from .statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)

class AIDetector:
    """Main orchestrator for comprehensive AI detection analysis"""
    
    def __init__(self):
        """Initialize all analysis components"""
        # Initialize analyzers
        self.text_analyzer = TextAIAnalyzer()
        self.image_analyzer = ImageAIAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.perplexity_analyzer = PerplexityAnalyzer()
        self.copyleaks_detector = CopyleaksDetector()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        logger.info("AI Detector initialized with all analyzers")
    
    def analyze_text(self, text: str, is_pro: bool = False) -> Dict[str, Any]:
        """
        Comprehensive text analysis for AI detection
        """
        try:
            # Start timing
            start_time = datetime.now()
            
            # Basic analysis (free tier)
            analysis_results = {
                'success': True,
                'content_type': 'text',
                'content_length': len(text),
                'word_count': len(text.split()),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'is_pro': is_pro
            }
            
            # Pattern analysis (always included)
            pattern_results = self.pattern_analyzer.analyze(text)
            analysis_results['pattern_analysis'] = pattern_results
            
            # Perplexity analysis (always included)
            perplexity_results = self.perplexity_analyzer.analyze(text)
            analysis_results['perplexity_analysis'] = perplexity_results
            
            # Statistical analysis (always included)
            statistical_results = self.statistical_analyzer.analyze(text)
            analysis_results['statistical_analysis'] = statistical_results
            
            # Calculate initial AI probability based on free analyses
            ai_probability = self._calculate_text_ai_probability(
                pattern_results, 
                perplexity_results,
                statistical_results
            )
            
            analysis_results['ai_probability'] = ai_probability
            analysis_results['confidence_level'] = self._get_confidence_level(ai_probability)
            
            # Pro features
            if is_pro:
                # Copyleaks AI detection
                copyleaks_results = self.copyleaks_detector.detect_ai(text)
                analysis_results['copyleaks_analysis'] = copyleaks_results
                
                # Advanced pattern detection
                analysis_results['advanced_patterns'] = self.pattern_analyzer.advanced_analysis(text)
                
                # Detailed linguistic analysis
                analysis_results['linguistic_analysis'] = self._perform_linguistic_analysis(text)
                
                # AI model fingerprinting
                analysis_results['model_detection'] = self._detect_ai_model(text)
                
                # Recalculate with all data
                ai_probability = self._calculate_pro_text_ai_probability(analysis_results)
                analysis_results['ai_probability'] = ai_probability
                analysis_results['confidence_level'] = self._get_confidence_level(ai_probability)
                
                # Add detailed breakdown
                analysis_results['detection_breakdown'] = self._create_detection_breakdown(analysis_results)
            
            # Add summary
            analysis_results['summary'] = self._create_text_summary(analysis_results)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            analysis_results['processing_time'] = round(processing_time, 2)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Text analysis error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'Text analysis failed: {str(e)}'
            }
    
    def analyze_image(self, image_data: str, image_type: str = 'image/jpeg', is_pro: bool = False) -> Dict[str, Any]:
        """
        Comprehensive image analysis for AI detection
        """
        try:
            # Start timing
            start_time = datetime.now()
            
            # Basic analysis
            analysis_results = {
                'success': True,
                'content_type': 'image',
                'image_type': image_type,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'is_pro': is_pro
            }
            
            # Image AI detection
            image_results = self.image_analyzer.analyze(image_data, image_type)
            analysis_results['image_analysis'] = image_results
            
            # Calculate AI probability
            ai_probability = image_results.get('ai_probability', 50)
            analysis_results['ai_probability'] = ai_probability
            analysis_results['confidence_level'] = self._get_confidence_level(ai_probability)
            
            # Pro features
            if is_pro:
                # Advanced image forensics
                analysis_results['forensics_analysis'] = self.image_analyzer.forensic_analysis(image_data)
                
                # AI model detection for images
                analysis_results['model_detection'] = self.image_analyzer.detect_generation_model(image_data)
                
                # Artifact detection
                analysis_results['artifact_analysis'] = self.image_analyzer.detect_ai_artifacts(image_data)
                
                # Recalculate with all data
                ai_probability = self._calculate_pro_image_ai_probability(analysis_results)
                analysis_results['ai_probability'] = ai_probability
                analysis_results['confidence_level'] = self._get_confidence_level(ai_probability)
                
                # Add detailed breakdown
                analysis_results['detection_breakdown'] = self._create_image_detection_breakdown(analysis_results)
            
            # Add summary
            analysis_results['summary'] = self._create_image_summary(analysis_results)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            analysis_results['processing_time'] = round(processing_time, 2)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'Image analysis failed: {str(e)}'
            }
    
    def analyze_url(self, url: str, is_pro: bool = False) -> Dict[str, Any]:
        """
        Analyze content from URL (text or image)
        """
        try:
            # First, extract content from URL
            content_data = self.text_analyzer.extract_from_url(url)
            
            if not content_data.get('success'):
                return content_data
            
            # Check if it's an image URL
            if self._is_image_url(url) or content_data.get('content_type', '').startswith('image/'):
                # Download and analyze image
                image_data = self._download_image(url)
                if image_data:
                    return self.analyze_image(image_data['data'], image_data['type'], is_pro)
                else:
                    return {'success': False, 'error': 'Failed to download image'}
            else:
                # Analyze text content
                text = content_data.get('text', '')
                if text:
                    result = self.analyze_text(text, is_pro)
                    # Add URL metadata
                    result['source_url'] = url
                    result['source_title'] = content_data.get('title')
                    result['source_domain'] = content_data.get('domain')
                    return result
                else:
                    return {'success': False, 'error': 'No text content found at URL'}
                    
        except Exception as e:
            logger.error(f"URL analysis error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'URL analysis failed: {str(e)}'
            }
    
    def _calculate_text_ai_probability(self, pattern_results: Dict, perplexity_results: Dict, 
                                     statistical_results: Dict) -> float:
        """Calculate AI probability for text based on free tier analyses"""
        scores = []
        weights = []
        
        # Pattern analysis score (weight: 30%)
        if pattern_results.get('ai_patterns_score') is not None:
            scores.append(pattern_results['ai_patterns_score'])
            weights.append(0.3)
        
        # Perplexity score (weight: 35%)
        if perplexity_results.get('ai_probability') is not None:
            scores.append(perplexity_results['ai_probability'])
            weights.append(0.35)
        
        # Statistical score (weight: 35%)
        if statistical_results.get('ai_probability') is not None:
            scores.append(statistical_results['ai_probability'])
            weights.append(0.35)
        
        if not scores:
            return 50.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        
        return round(weighted_sum / total_weight, 1)
    
    def _calculate_pro_text_ai_probability(self, analysis_results: Dict) -> float:
        """Calculate AI probability with all pro features"""
        scores = []
        weights = []
        
        # Basic scores (40% total)
        if analysis_results.get('pattern_analysis', {}).get('ai_patterns_score') is not None:
            scores.append(analysis_results['pattern_analysis']['ai_patterns_score'])
            weights.append(0.15)
        
        if analysis_results.get('perplexity_analysis', {}).get('ai_probability') is not None:
            scores.append(analysis_results['perplexity_analysis']['ai_probability'])
            weights.append(0.15)
        
        if analysis_results.get('statistical_analysis', {}).get('ai_probability') is not None:
            scores.append(analysis_results['statistical_analysis']['ai_probability'])
            weights.append(0.10)
        
        # Copyleaks score (40% - highest weight due to accuracy)
        if analysis_results.get('copyleaks_analysis', {}).get('ai_probability') is not None:
            scores.append(analysis_results['copyleaks_analysis']['ai_probability'])
            weights.append(0.40)
        
        # Advanced patterns (10%)
        if analysis_results.get('advanced_patterns', {}).get('ai_score') is not None:
            scores.append(analysis_results['advanced_patterns']['ai_score'])
            weights.append(0.10)
        
        # Model detection (10%)
        if analysis_results.get('model_detection', {}).get('confidence') is not None:
            scores.append(analysis_results['model_detection']['confidence'])
            weights.append(0.10)
        
        if not scores:
            return 50.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        
        return round(weighted_sum / total_weight, 1)
    
    def _calculate_pro_image_ai_probability(self, analysis_results: Dict) -> float:
        """Calculate AI probability for images with pro features"""
        scores = []
        weights = []
        
        # Basic image analysis (40%)
        if analysis_results.get('image_analysis', {}).get('ai_probability') is not None:
            scores.append(analysis_results['image_analysis']['ai_probability'])
            weights.append(0.40)
        
        # Forensics analysis (30%)
        if analysis_results.get('forensics_analysis', {}).get('ai_probability') is not None:
            scores.append(analysis_results['forensics_analysis']['ai_probability'])
            weights.append(0.30)
        
        # Model detection (20%)
        if analysis_results.get('model_detection', {}).get('confidence') is not None:
            scores.append(analysis_results['model_detection']['confidence'])
            weights.append(0.20)
        
        # Artifact analysis (10%)
        if analysis_results.get('artifact_analysis', {}).get('ai_score') is not None:
            scores.append(analysis_results['artifact_analysis']['ai_score'])
            weights.append(0.10)
        
        if not scores:
            return 50.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        
        return round(weighted_sum / total_weight, 1)
    
    def _get_confidence_level(self, ai_probability: float) -> str:
        """Get confidence level description"""
        if ai_probability >= 90:
            return "Very High"
        elif ai_probability >= 75:
            return "High"
        elif ai_probability >= 60:
            return "Moderate"
        elif ai_probability >= 40:
            return "Low"
        else:
            return "Very Low"
    
    def _create_text_summary(self, analysis_results: Dict) -> str:
        """Create a summary for text analysis"""
        ai_prob = analysis_results.get('ai_probability', 50)
        confidence = analysis_results.get('confidence_level', 'Unknown')
        
        if ai_prob >= 80:
            summary = f"This text shows strong indicators of AI generation with {ai_prob}% probability. "
            summary += "Multiple detection methods confirm AI involvement."
        elif ai_prob >= 60:
            summary = f"This text likely contains AI-generated content with {ai_prob}% probability. "
            summary += "Several AI patterns were detected."
        elif ai_prob >= 40:
            summary = f"This text shows some AI characteristics with {ai_prob}% probability. "
            summary += "Mixed signals suggest possible AI assistance or editing."
        else:
            summary = f"This text appears to be human-written with only {ai_prob}% AI probability. "
            summary += "Natural writing patterns dominate."
        
        return summary
    
    def _create_image_summary(self, analysis_results: Dict) -> str:
        """Create a summary for image analysis"""
        ai_prob = analysis_results.get('ai_probability', 50)
        
        if ai_prob >= 80:
            summary = f"This image is very likely AI-generated with {ai_prob}% probability. "
            summary += "Strong AI generation artifacts detected."
        elif ai_prob >= 60:
            summary = f"This image shows significant AI generation indicators with {ai_prob}% probability. "
            summary += "Multiple AI characteristics present."
        elif ai_prob >= 40:
            summary = f"This image has some AI generation markers with {ai_prob}% probability. "
            summary += "Mixed indicators suggest possible AI involvement."
        else:
            summary = f"This image appears to be authentic with only {ai_prob}% AI probability. "
            summary += "Natural image characteristics dominate."
        
        return summary
    
    def _create_detection_breakdown(self, analysis_results: Dict) -> Dict[str, Any]:
        """Create detailed breakdown of detection results"""
        breakdown = {
            'detection_methods': [],
            'key_indicators': [],
            'confidence_factors': []
        }
        
        # Add detection methods
        if 'pattern_analysis' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Pattern Analysis',
                'score': analysis_results['pattern_analysis'].get('ai_patterns_score', 0),
                'weight': '15%'
            })
        
        if 'perplexity_analysis' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Perplexity Analysis',
                'score': analysis_results['perplexity_analysis'].get('ai_probability', 0),
                'weight': '15%'
            })
        
        if 'statistical_analysis' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Statistical Analysis',
                'score': analysis_results['statistical_analysis'].get('ai_probability', 0),
                'weight': '10%'
            })
        
        if 'copyleaks_analysis' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Copyleaks AI Detection',
                'score': analysis_results['copyleaks_analysis'].get('ai_probability', 0),
                'weight': '40%'
            })
        
        # Add key indicators
        if analysis_results.get('pattern_analysis', {}).get('detected_patterns'):
            for pattern in analysis_results['pattern_analysis']['detected_patterns']:
                breakdown['key_indicators'].append({
                    'type': 'pattern',
                    'description': pattern['description'],
                    'severity': pattern['severity']
                })
        
        # Add confidence factors
        if analysis_results.get('confidence_level') == 'Very High':
            breakdown['confidence_factors'].append('Multiple methods show strong agreement')
        elif analysis_results.get('confidence_level') == 'High':
            breakdown['confidence_factors'].append('Most methods indicate AI generation')
        
        if analysis_results.get('copyleaks_analysis', {}).get('confidence') == 'high':
            breakdown['confidence_factors'].append('Professional AI detection confirms findings')
        
        return breakdown
    
    def _create_image_detection_breakdown(self, analysis_results: Dict) -> Dict[str, Any]:
        """Create detailed breakdown for image detection"""
        breakdown = {
            'detection_methods': [],
            'visual_indicators': [],
            'technical_markers': []
        }
        
        # Add methods and scores
        if 'image_analysis' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Visual AI Detection',
                'score': analysis_results['image_analysis'].get('ai_probability', 0),
                'weight': '40%'
            })
        
        if 'forensics_analysis' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Digital Forensics',
                'score': analysis_results['forensics_analysis'].get('ai_probability', 0),
                'weight': '30%'
            })
        
        if 'model_detection' in analysis_results:
            breakdown['detection_methods'].append({
                'method': 'Model Detection',
                'score': analysis_results['model_detection'].get('confidence', 0),
                'weight': '20%'
            })
        
        # Add visual indicators
        if analysis_results.get('image_analysis', {}).get('artifact_analysis', {}).get('artifacts_found'):
            for artifact in analysis_results['image_analysis']['artifact_analysis']['artifacts_found']:
                breakdown['visual_indicators'].append(artifact)
        
        # Add technical markers
        if analysis_results.get('forensics_analysis', {}).get('artifacts_detected'):
            for marker in analysis_results['forensics_analysis']['artifacts_detected']:
                breakdown['technical_markers'].append(marker)
        
        return breakdown
    
    def _perform_linguistic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform advanced linguistic analysis"""
        try:
            # Calculate various linguistic metrics
            words = text.split()
            sentences = self._split_sentences(text)
            
            # Vocabulary complexity
            unique_words = set(word.lower() for word in words)
            vocabulary_complexity = 'high' if len(unique_words) / len(words) > 0.5 else 'moderate'
            
            # Sentence variety
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = self._calculate_variance(sentence_lengths) if sentence_lengths else 0
            sentence_variety = 'high' if length_variance > 50 else 'low'
            
            # Coherence score (simplified)
            coherence_score = 85 if len(sentences) > 5 else 75
            
            # Style consistency (simplified)
            style_consistency = 92 if len(unique_words) > 100 else 85
            
            return {
                'vocabulary_complexity': vocabulary_complexity,
                'sentence_variety': sentence_variety,
                'coherence_score': coherence_score,
                'style_consistency': style_consistency,
                'avg_sentence_length': round(sum(sentence_lengths) / len(sentence_lengths), 1) if sentence_lengths else 0,
                'unique_word_ratio': round(len(unique_words) / len(words), 3) if words else 0
            }
        except Exception as e:
            logger.error(f"Linguistic analysis error: {e}")
            return {
                'vocabulary_complexity': 'moderate',
                'sentence_variety': 'low',
                'coherence_score': 85,
                'style_consistency': 92
            }
    
    def _detect_ai_model(self, text: str) -> Dict[str, Any]:
        """Attempt to detect which AI model generated the text"""
        try:
            # Model fingerprinting based on patterns
            detected_model = 'Unknown'
            confidence = 50
            model_signatures = []
            
            text_lower = text.lower()
            
            # GPT signatures
            gpt_patterns = [
                ('I cannot and will not', 'GPT Safety Response'),
                ('As an AI language model', 'GPT Self-Reference'),
                ('I don\'t have access to real-time', 'GPT Limitation Statement'),
                ('It\'s important to note that', 'GPT Hedging Pattern'),
                ('I must emphasize', 'GPT Emphasis Pattern'),
                ('I should clarify', 'GPT Clarification Pattern')
            ]
            
            for pattern, name in gpt_patterns:
                if pattern.lower() in text_lower:
                    model_signatures.append({
                        'model': 'GPT',
                        'signature': name,
                        'confidence': 85
                    })
                    detected_model = 'GPT (ChatGPT/GPT-4)'
                    confidence = max(confidence, 85)
            
            # Claude signatures
            claude_patterns = [
                ('I understand you\'re asking', 'Claude Understanding Pattern'),
                ('I\'d be happy to help', 'Claude Helpful Pattern'),
                ('Could you clarify', 'Claude Clarification Pattern'),
                ('I appreciate your', 'Claude Appreciation Pattern'),
                ('Let me address', 'Claude Structure Pattern')
            ]
            
            for pattern, name in claude_patterns:
                if pattern.lower() in text_lower:
                    model_signatures.append({
                        'model': 'Claude',
                        'signature': name,
                        'confidence': 80
                    })
                    if detected_model == 'Unknown':
                        detected_model = 'Claude (Anthropic)'
                        confidence = max(confidence, 80)
            
            # Bard/Gemini signatures
            bard_patterns = [
                ('based on my knowledge', 'Bard Knowledge Pattern'),
                ('I can help you with', 'Bard Assistance Pattern'),
                ('Here\'s what I found', 'Bard Search Pattern')
            ]
            
            for pattern, name in bard_patterns:
                if pattern.lower() in text_lower:
                    model_signatures.append({
                        'model': 'Bard/Gemini',
                        'signature': name,
                        'confidence': 75
                    })
                    if detected_model == 'Unknown':
                        detected_model = 'Bard/Gemini (Google)'
                        confidence = max(confidence, 75)
            
            return {
                'detected_model': detected_model,
                'confidence': confidence,
                'model_signatures': model_signatures
            }
            
        except Exception as e:
            logger.error(f"Model detection error: {e}")
            return {
                'detected_model': 'Unknown',
                'confidence': 50,
                'model_signatures': []
            }
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico']
        url_lower = url.lower()
        return any(url_lower.endswith(ext) for ext in image_extensions)
    
    def _download_image(self, url: str) -> Optional[Dict[str, Any]]:
        """Download image from URL and convert to base64"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                image_data = base64.b64encode(response.content).decode('utf-8')
                content_type = response.headers.get('content-type', 'image/jpeg')
                
                return {
                    'data': image_data,
                    'type': content_type
                }
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
        
        return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
