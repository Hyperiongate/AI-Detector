"""
Perplexity Analyzer Service
Analyzes text perplexity and burstiness for AI detection
"""
import math
import logging
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)

class PerplexityAnalyzer:
    """Analyzes text perplexity and related metrics"""
    
    def __init__(self):
        # Common words that shouldn't heavily influence perplexity
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from'
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text perplexity and burstiness"""
        try:
            # Tokenize text
            words = text.lower().split()
            sentences = self._split_sentences(text)
            
            if len(words) < 10:
                return {
                    'perplexity': 50,
                    'burstiness': 0.5,
                    'ai_probability': 50,
                    'error': 'Text too short for analysis'
                }
            
            # Calculate perplexity
            perplexity = self._calculate_perplexity(words)
            
            # Calculate burstiness
            burstiness = self._calculate_burstiness(sentences)
            
            # Calculate sentence length variance
            sentence_variance = self._calculate_sentence_variance(sentences)
            
            # Calculate vocabulary richness
            vocabulary_richness = self._calculate_vocabulary_richness(words)
            
            # Calculate AI probability based on metrics
            ai_probability = self._calculate_ai_probability(
                perplexity, burstiness, sentence_variance, vocabulary_richness
            )
            
            return {
                'perplexity': round(perplexity, 2),
                'burstiness': round(burstiness, 3),
                'sentence_variance': round(sentence_variance, 2),
                'vocabulary_richness': round(vocabulary_richness, 3),
                'ai_probability': round(ai_probability, 1),
                'interpretation': self._interpret_results(perplexity, burstiness),
                'metrics': {
                    'word_count': len(words),
                    'sentence_count': len(sentences),
                    'avg_sentence_length': round(len(words) / len(sentences), 1) if sentences else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Perplexity analysis error: {str(e)}", exc_info=True)
            return {
                'perplexity': 50,
                'burstiness': 0.5,
                'ai_probability': 50,
                'error': str(e)
            }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_perplexity(self, words: List[str]) -> float:
        """
        Calculate perplexity - lower values indicate more predictable (AI-like) text
        """
        # Build word frequency distribution
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate word probabilities
        probabilities = []
        for i in range(1, len(words)):
            current_word = words[i]
            prev_word = words[i-1]
            
            # Simple bigram probability
            bigram_count = sum(1 for j in range(len(words)-1) 
                             if words[j] == prev_word and words[j+1] == current_word)
            prev_count = word_freq[prev_word]
            
            # Smoothed probability
            prob = (bigram_count + 1) / (prev_count + len(set(words)))
            probabilities.append(prob)
        
        # Calculate perplexity
        if not probabilities:
            return 50
        
        # Compute geometric mean of inverse probabilities
        log_sum = sum(math.log(1/p) for p in probabilities if p > 0)
        perplexity = math.exp(log_sum / len(probabilities))
        
        # Normalize to 0-100 scale
        # Typical human text: 50-200, AI text: 10-50
        normalized = min(100, max(0, (perplexity / 2)))
        
        return normalized
    
    def _calculate_burstiness(self, sentences: List[str]) -> float:
        """
        Calculate burstiness - variation in sentence lengths
        Human text is more 'bursty' with varied sentence lengths
        """
        if len(sentences) < 2:
            return 0.5
        
        # Get sentence lengths
        lengths = [len(s.split()) for s in sentences]
        
        # Calculate standard deviation
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # Calculate burstiness score (coefficient of variation)
        burstiness = std_dev / mean_length if mean_length > 0 else 0
        
        # Normalize to 0-1 scale
        return min(1, burstiness)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance without numpy"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_sentence_variance(self, sentences: List[str]) -> float:
        """Calculate variance in sentence structures"""
        if len(sentences) < 2:
            return 0
        
        # Analyze sentence starters
        starters = [s.split()[0].lower() for s in sentences if s.split()]
        unique_starters = len(set(starters))
        starter_diversity = unique_starters / len(starters) if starters else 0
        
        # Analyze sentence lengths
        lengths = [len(s.split()) for s in sentences]
        length_variance = self._calculate_variance(lengths) if lengths else 0
        
        # Combined score
        return (starter_diversity * 50) + min(50, length_variance / 2)
    
    def _calculate_vocabulary_richness(self, words: List[str]) -> float:
        """Calculate vocabulary diversity"""
        # Remove common words for this analysis
        content_words = [w for w in words if w not in self.common_words]
        
        if not content_words:
            return 0.5
        
        unique_words = set(content_words)
        richness = len(unique_words) / len(content_words)
        
        return richness
    
    def _calculate_ai_probability(self, perplexity: float, burstiness: float, 
                                 sentence_variance: float, vocabulary_richness: float) -> float:
        """Calculate overall AI probability based on metrics"""
        # Lower perplexity = more AI-like
        perplexity_score = 100 - perplexity
        
        # Lower burstiness = more AI-like
        burstiness_score = (1 - burstiness) * 100
        
        # Lower variance = more AI-like
        variance_score = 100 - sentence_variance
        
        # Lower vocabulary richness = more AI-like (for shorter texts)
        richness_score = (1 - vocabulary_richness) * 100 if vocabulary_richness < 0.4 else 0
        
        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # Perplexity is most important
        scores = [perplexity_score, burstiness_score, variance_score, richness_score]
        
        ai_probability = sum(s * w for s, w in zip(scores, weights))
        
        return max(0, min(100, ai_probability))
    
    def _interpret_results(self, perplexity: float, burstiness: float) -> str:
        """Provide interpretation of the results"""
        if perplexity < 30 and burstiness < 0.3:
            return "Strong indicators of AI generation: very low perplexity and minimal sentence variation"
        elif perplexity < 50 and burstiness < 0.5:
            return "Likely AI-generated or heavily AI-edited: predictable patterns and uniform structure"
        elif perplexity > 70 and burstiness > 0.7:
            return "Likely human-written: high perplexity and natural sentence variation"
        else:
            return "Mixed signals: possibly AI-assisted or edited content"
