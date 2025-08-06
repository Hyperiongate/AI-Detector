"""
AI Detector Service
Main orchestrator for AI detection analysis
"""
import os
import logging
from typing import Dict, Any, Optional, List
import requests
from datetime import datetime

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
        
        return breakdown
    
    def _perform_linguistic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform advanced linguistic analysis"""
        # This would include more sophisticated NLP analysis
        return {
            'vocabulary_complexity': 'moderate',
            'sentence_variety': 'low',
            'coherence_score': 85,
            'style_consistency': 92
        }
    
    def _detect_ai_model(self, text: str) -> Dict[str, Any]:
        """Attempt to detect which AI model generated the text"""
        # This would use model fingerprinting techniques
        return {
            'detected_model': 'Unknown',
            'confidence': 50,
            'model_signatures': []
        }
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        return any(url.lower().endswith(ext) for ext in image_extensions)
    
    def _download_image(self, url: str) -> Optional[Dict[str, Any]]:
        """Download image from URL and convert to base64"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                import base64
                image_data = base64.b64encode(response.content).decode('utf-8')
                content_type = response.headers.get('content-type', 'image/jpeg')
                return {
                    'data': image_data,
                    'type': content_type
                }
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
        return None
