"""
Statistical Analyzer Service
Performs statistical analysis on text for AI detection
"""
import logging
from typing import Dict, Any, List
from collections import Counter
import re
import math

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Analyzes statistical properties of text"""
    
    def __init__(self):
        # Initialize with common statistical thresholds
        self.ai_thresholds = {
            'word_length_std': 2.5,  # AI text has more consistent word lengths
            'sentence_length_std': 8.0,  # AI text has more consistent sentence lengths
            'punctuation_rate': 0.12,  # AI text has predictable punctuation
            'unique_word_ratio': 0.45,  # AI text may have lower vocabulary diversity
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            # Basic preprocessing
            words = text.split()
            sentences = self._split_sentences(text)
            
            if len(words) < 20:
                return {
                    'ai_probability': 50,
                    'error': 'Text too short for statistical analysis'
                }
            
            # Calculate various statistics
            word_stats = self._analyze_word_statistics(words)
            sentence_stats = self._analyze_sentence_statistics(sentences)
            char_stats = self._analyze_character_statistics(text)
            lexical_stats = self._analyze_lexical_diversity(words)
            
            # Calculate AI probability
            ai_probability = self._calculate_ai_probability(
                word_stats, sentence_stats, char_stats, lexical_stats
            )
            
            return {
                'ai_probability': round(ai_probability, 1),
                'word_statistics': word_stats,
                'sentence_statistics': sentence_stats,
                'character_statistics': char_stats,
                'lexical_diversity': lexical_stats,
                'vocabulary_diversity': lexical_stats.get('type_token_ratio', 0),
                'sentence_variance': sentence_stats.get('length_variance', 0),
                'interpretation': self._interpret_results(ai_probability, word_stats, sentence_stats)
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis error: {str(e)}", exc_info=True)
            return {
                'ai_probability': 50,
                'error': str(e)
            }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_word_statistics(self, words: List[str]) -> Dict[str, Any]:
        """Analyze word-level statistics"""
        word_lengths = [len(word) for word in words]
        
        # Calculate statistics
        avg_length = sum(word_lengths) / len(word_lengths)
        variance = sum((l - avg_length) ** 2 for l in word_lengths) / len(word_lengths)
        std_dev = math.sqrt(variance)
        
        # Word frequency distribution
        word_freq = Counter(words)
        
        # Hapax legomena (words appearing only once)
        hapax_count = sum(1 for word, count in word_freq.items() if count == 1)
        hapax_ratio = hapax_count / len(words)
        
        return {
            'avg_word_length': round(avg_length, 2),
            'word_length_std': round(std_dev, 2),
            'word_length_variance': round(variance, 2),
            'hapax_legomena_ratio': round(hapax_ratio, 3),
            'total_words': len(words),
            'unique_words': len(set(words))
        }
    
    def _analyze_sentence_statistics(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentence-level statistics"""
        sentence_lengths = [len(s.split()) for s in sentences]
        
        if not sentence_lengths:
            return {}
        
        # Calculate statistics
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        std_dev = math.sqrt(variance)
        
        # Sentence starter analysis
        starters = [s.split()[0].lower() for s in sentences if s.split()]
        unique_starters = len(set(starters))
        starter_diversity = unique_starters / len(starters) if starters else 0
        
        return {
            'avg_sentence_length': round(avg_length, 2),
            'sentence_length_std': round(std_dev, 2),
            'length_variance': round(variance, 2),
            'total_sentences': len(sentences),
            'starter_diversity': round(starter_diversity, 3),
            'min_length': min(sentence_lengths),
            'max_length': max(sentence_lengths)
        }
    
    def _analyze_character_statistics(self, text: str) -> Dict[str, Any]:
        """Analyze character-level statistics"""
        # Character frequency
        char_freq = Counter(text.lower())
        total_chars = len(text)
        
        # Punctuation analysis
        punctuation_chars = sum(1 for c in text if c in '.,!?;:-"\' ')
        punctuation_rate = punctuation_chars / total_chars
        
        # Digit analysis
        digit_count = sum(1 for c in text if c.isdigit())
        digit_rate = digit_count / total_chars
        
        # Capital letter analysis
        capital_count = sum(1 for c in text if c.isupper())
        capital_rate = capital_count / total_chars
        
        return {
            'total_characters': total_chars,
            'punctuation_rate': round(punctuation_rate, 3),
            'digit_rate': round(digit_rate, 3),
            'capital_rate': round(capital_rate, 3),
            'space_rate': round(text.count(' ') / total_chars, 3)
        }
    
    def _analyze_lexical_diversity(self, words: List[str]) -> Dict[str, Any]:
        """Analyze lexical diversity and richness"""
        # Type-token ratio
        unique_words = set(words)
        type_token_ratio = len(unique_words) / len(words)
        
        # Yule's K statistic (vocabulary richness)
        word_freq = Counter(words)
        M1 = len(words)
        M2 = sum(freq ** 2 for freq in word_freq.values())
        yules_k = 10000 * (M2 - M1) / (M1 ** 2) if M1 > 0 else 0
        
        # Simpson's diversity index
        N = len(words)
        simpsons_d = sum(n * (n - 1) for n in word_freq.values()) / (N * (N - 1)) if N > 1 else 0
        
        return {
            'type_token_ratio': round(type_token_ratio, 3),
            'yules_k': round(yules_k, 2),
            'simpsons_diversity': round(1 - simpsons_d, 3),
            'vocabulary_size': len(unique_words)
        }
    
    def _calculate_ai_probability(self, word_stats: Dict, sentence_stats: Dict, 
                                 char_stats: Dict, lexical_stats: Dict) -> float:
        """Calculate AI probability based on statistical metrics"""
        score = 0
        weights = 0
        
        # Word length consistency (AI text is more consistent)
        if 'word_length_std' in word_stats:
            std_dev = word_stats['word_length_std']
            if std_dev < self.ai_thresholds['word_length_std']:
                word_score = (self.ai_thresholds['word_length_std'] - std_dev) / self.ai_thresholds['word_length_std'] * 100
                score += word_score * 0.2
                weights += 0.2
        
        # Sentence length consistency
        if 'sentence_length_std' in sentence_stats:
            std_dev = sentence_stats['sentence_length_std']
            if std_dev < self.ai_thresholds['sentence_length_std']:
                sent_score = (self.ai_thresholds['sentence_length_std'] - std_dev) / self.ai_thresholds['sentence_length_std'] * 100
                score += sent_score * 0.25
                weights += 0.25
        
        # Punctuation rate (AI text has predictable punctuation)
        if 'punctuation_rate' in char_stats:
            punct_rate = char_stats['punctuation_rate']
            if abs(punct_rate - self.ai_thresholds['punctuation_rate']) < 0.02:
                punct_score = 80
                score += punct_score * 0.15
                weights += 0.15
        
        # Vocabulary diversity (AI text may have lower diversity)
        if 'type_token_ratio' in lexical_stats:
            ttr = lexical_stats['type_token_ratio']
            if ttr < self.ai_thresholds['unique_word_ratio']:
                diversity_score = (self.ai_thresholds['unique_word_ratio'] - ttr) / self.ai_thresholds['unique_word_ratio'] * 100
                score += diversity_score * 0.2
                weights += 0.2
        
        # Starter diversity (AI text has less diverse sentence starters)
        if 'starter_diversity' in sentence_stats:
            starter_div = sentence_stats['starter_diversity']
            if starter_div < 0.5:
                starter_score = (0.5 - starter_div) * 200
                score += starter_score * 0.2
                weights += 0.2
        
        # Normalize score
        if weights > 0:
            final_score = score / weights
        else:
            final_score = 50
        
        return max(0, min(100, final_score))
    
    def _interpret_results(self, ai_probability: float, word_stats: Dict, sentence_stats: Dict) -> str:
        """Provide interpretation of statistical results"""
        interpretations = []
        
        if ai_probability >= 70:
            interpretations.append("High statistical indicators of AI generation")
        elif ai_probability >= 50:
            interpretations.append("Moderate statistical indicators suggesting possible AI involvement")
        else:
            interpretations.append("Statistical patterns consistent with human writing")
        
        # Add specific observations
        if word_stats.get('word_length_std', 10) < 2.5:
            interpretations.append("unusually consistent word lengths")
        
        if sentence_stats.get('sentence_length_std', 10) < 8.0:
            interpretations.append("very uniform sentence structure")
        
        if sentence_stats.get('starter_diversity', 1) < 0.3:
            interpretations.append("repetitive sentence beginnings")
        
        return ". ".join(interpretations) if interpretations else "No significant patterns detected"
