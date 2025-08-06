"""
Pattern Analyzer Service
Detects AI-specific writing patterns and behaviors
"""
import re
import logging
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analyzes text for AI-specific patterns"""
    
    def __init__(self):
        # Common AI patterns and phrases
        self.ai_indicators = {
            'formal_transitions': [
                'furthermore', 'moreover', 'nevertheless', 'consequently',
                'therefore', 'hence', 'thus', 'accordingly', 'subsequently'
            ],
            'hedging_phrases': [
                'it is important to note', 'it\'s worth noting', 'it should be noted',
                'one might argue', 'it could be said', 'arguably', 'potentially',
                'it is worth mentioning', 'it\'s crucial to understand'
            ],
            'ai_favorites': [
                'delve into', 'tapestry', 'landscape', 'realm', 'navigate',
                'leverage', 'utilize', 'implement', 'facilitate', 'optimize',
                'enhance', 'streamline', 'bolster', 'underscore', 'pivotal',
                'paramount', 'multifaceted', 'comprehensive', 'nuanced'
            ],
            'conclusion_starters': [
                'in conclusion', 'to conclude', 'in summary', 'to summarize',
                'all in all', 'overall', 'in essence', 'ultimately'
            ],
            'list_indicators': [
                'firstly', 'secondly', 'thirdly', 'lastly', 'finally',
                'first of all', 'to begin with', 'in addition', 'additionally'
            ]
        }
        
        # AI-specific sentence patterns
        self.sentence_patterns = {
            'perfect_grammar': re.compile(r'^[A-Z][^.!?]*[.!?]$'),
            'balanced_structure': re.compile(r'(not only|both).*(but also|and)'),
            'nested_clauses': re.compile(r'(which|that|who|whom|whose).*,.*[.!?]'),
            'passive_voice': re.compile(r'(is|are|was|were|been|being) \w+ed\b')
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for AI patterns"""
        try:
            # Clean text
            text_lower = text.lower()
            sentences = self._split_sentences(text)
            
            # Pattern detection
            detected_patterns = []
            pattern_scores = {}
            
            # Check for AI indicators
            for category, phrases in self.ai_indicators.items():
                count = sum(1 for phrase in phrases if phrase in text_lower)
                if count > 0:
                    severity = self._calculate_severity(count, len(sentences))
                    pattern_scores[category] = severity
                    detected_patterns.append({
                        'type': category,
                        'count': count,
                        'severity': severity,
                        'description': self._get_pattern_description(category)
                    })
            
            # Analyze sentence patterns
            pattern_analysis = self._analyze_sentence_patterns(sentences)
            
            # Calculate overall AI pattern score
            ai_patterns_score = self._calculate_ai_score(pattern_scores, pattern_analysis)
            
            return {
                'ai_patterns_score': ai_patterns_score,
                'detected_patterns': detected_patterns,
                'pattern_analysis': pattern_analysis,
                'indicator_counts': {
                    category: sum(1 for phrase in phrases if phrase in text_lower)
                    for category, phrases in self.ai_indicators.items()
                },
                'total_indicators': sum(
                    sum(1 for phrase in phrases if phrase in text_lower)
                    for phrases in self.ai_indicators.values()
                )
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {str(e)}", exc_info=True)
            return {
                'ai_patterns_score': 50,
                'error': str(e)
            }
    
    def advanced_analysis(self, text: str) -> Dict[str, Any]:
        """Advanced pattern analysis for pro users"""
        sentences = self._split_sentences(text)
        
        # Paragraph structure analysis
        paragraphs = text.split('\n\n')
        paragraph_analysis = self._analyze_paragraph_structure(paragraphs)
        
        # Repetition analysis
        repetition_analysis = self._analyze_repetitions(text)
        
        # Formulaic structure detection
        formulaic_score = self._detect_formulaic_structure(sentences)
        
        # AI model fingerprints
        model_signatures = self._detect_model_signatures(text)
        
        return {
            'paragraph_analysis': paragraph_analysis,
            'repetition_analysis': repetition_analysis,
            'formulaic_score': formulaic_score,
            'model_signatures': model_signatures,
            'ai_score': self._calculate_advanced_ai_score({
                'paragraph': paragraph_analysis.get('ai_score', 50),
                'repetition': repetition_analysis.get('ai_score', 50),
                'formulaic': formulaic_score,
                'signatures': len(model_signatures) * 20
            })
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence_patterns(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentence-level patterns"""
        if not sentences:
            return {}
        
        # Sentence length consistency
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Pattern matching
        pattern_counts = {
            pattern_name: sum(1 for s in sentences if pattern.search(s))
            for pattern_name, pattern in self.sentence_patterns.items()
        }
        
        # Sentence starter analysis
        starters = [s.split()[0].lower() for s in sentences if s.split()]
        starter_diversity = len(set(starters)) / len(starters) if starters else 0
        
        return {
            'avg_sentence_length': round(avg_length, 1),
            'length_variance': round(length_variance, 2),
            'length_consistency': 1 - min(length_variance / avg_length, 1) if avg_length > 0 else 0,
            'pattern_counts': pattern_counts,
            'starter_diversity': round(starter_diversity, 2),
            'perfect_grammar_rate': pattern_counts.get('perfect_grammar', 0) / len(sentences)
        }
    
    def _analyze_paragraph_structure(self, paragraphs: List[str]) -> Dict[str, Any]:
        """Analyze paragraph-level structure"""
        if not paragraphs:
            return {'ai_score': 50}
        
        # Paragraph lengths
        lengths = [len(p.split()) for p in paragraphs if p.strip()]
        if not lengths:
            return {'ai_score': 50}
        
        avg_length = sum(lengths) / len(lengths)
        
        # Check for consistent paragraph structure
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        consistency_score = 1 - min(length_variance / avg_length, 1) if avg_length > 0 else 0
        
        # Topic sentence analysis
        topic_sentence_score = self._analyze_topic_sentences(paragraphs)
        
        ai_score = (consistency_score * 40 + topic_sentence_score * 60)
        
        return {
            'avg_paragraph_length': round(avg_length, 1),
            'consistency_score': round(consistency_score, 2),
            'topic_sentence_score': round(topic_sentence_score, 2),
            'ai_score': round(ai_score, 1)
        }
    
    def _analyze_topic_sentences(self, paragraphs: List[str]) -> float:
        """Analyze topic sentence patterns"""
        topic_patterns = [
            r'^(This|These|The|In|On|For|By|With|Through)',
            r'^[A-Z]\w+ing\b',  # Gerund starts
            r'^(However|Moreover|Furthermore|Additionally)'
        ]
        
        pattern_count = 0
        valid_paragraphs = [p for p in paragraphs if p.strip()]
        
        for para in valid_paragraphs:
            first_sentence = para.split('.')[0] if '.' in para else para
            for pattern in topic_patterns:
                if re.match(pattern, first_sentence.strip()):
                    pattern_count += 1
                    break
        
        return (pattern_count / len(valid_paragraphs) * 100) if valid_paragraphs else 0
    
    def _analyze_repetitions(self, text: str) -> Dict[str, Any]:
        """Analyze repetitive patterns"""
        words = text.lower().split()
        
        # N-gram analysis
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        # Count repetitions
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        # Find overused phrases
        overused_bigrams = {bg: count for bg, count in bigram_counts.items() if count > 2}
        overused_trigrams = {tg: count for tg, count in trigram_counts.items() if count > 2}
        
        # Calculate AI score based on repetition
        repetition_rate = (len(overused_bigrams) + len(overused_trigrams)) / (len(bigrams) + len(trigrams))
        ai_score = min(repetition_rate * 500, 100)  # High repetition suggests AI
        
        return {
            'overused_bigrams': overused_bigrams,
            'overused_trigrams': overused_trigrams,
            'repetition_rate': round(repetition_rate, 3),
            'ai_score': round(ai_score, 1)
        }
    
    def _detect_formulaic_structure(self, sentences: List[str]) -> float:
        """Detect formulaic writing structure"""
        if len(sentences) < 3:
            return 50
        
        # Check for common AI formulas
        formulas = {
            'intro_body_conclusion': self._check_intro_body_conclusion(sentences),
            'point_evidence_explain': self._check_point_evidence_pattern(sentences),
            'claim_support_claim': self._check_claim_pattern(sentences)
        }
        
        # Calculate score
        detected_formulas = sum(1 for score in formulas.values() if score > 70)
        return min(detected_formulas * 30 + 20, 100)
    
    def _check_intro_body_conclusion(self, sentences: List[str]) -> float:
        """Check for intro-body-conclusion pattern"""
        if len(sentences) < 5:
            return 0
        
        # Check first sentence for introduction patterns
        intro_patterns = ['this article', 'this essay', 'this paper', 'we will', 'i will']
        has_intro = any(pattern in sentences[0].lower() for pattern in intro_patterns)
        
        # Check last sentences for conclusion patterns
        conclusion_patterns = ['in conclusion', 'to conclude', 'in summary', 'overall']
        has_conclusion = any(
            pattern in sent.lower() 
            for sent in sentences[-3:] 
            for pattern in conclusion_patterns
        )
        
        return 100 if has_intro and has_conclusion else 0
    
    def _check_point_evidence_pattern(self, sentences: List[str]) -> float:
        """Check for point-evidence-explanation pattern"""
        evidence_words = ['for example', 'for instance', 'such as', 'specifically', 'research shows']
        explanation_words = ['this shows', 'this demonstrates', 'this suggests', 'therefore']
        
        pattern_count = 0
        for i in range(len(sentences) - 2):
            has_evidence = any(word in sentences[i+1].lower() for word in evidence_words)
            has_explanation = any(word in sentences[i+2].lower() for word in explanation_words)
            if has_evidence and has_explanation:
                pattern_count += 1
        
        return min((pattern_count / max(len(sentences) - 2, 1)) * 200, 100)
    
    def _check_claim_pattern(self, sentences: List[str]) -> float:
        """Check for claim-support-claim pattern"""
        claim_starters = ['it is', 'this is', 'there is', 'research', 'studies']
        support_words = ['because', 'since', 'as', 'due to', 'given that']
        
        pattern_count = 0
        for i in range(0, len(sentences) - 2, 3):
            has_claim = any(starter in sentences[i].lower() for starter in claim_starters)
            has_support = any(word in sentences[i+1].lower() for word in support_words)
            if has_claim and has_support:
                pattern_count += 1
        
        return min((pattern_count / max(len(sentences) // 3, 1)) * 150, 100)
    
    def _detect_model_signatures(self, text: str) -> List[Dict[str, Any]]:
        """Detect specific AI model signatures"""
        signatures = []
        
        # GPT signatures
        gpt_patterns = [
            (r'I cannot and will not', 'GPT Safety Response'),
            (r'As an AI language model', 'GPT Self-Reference'),
            (r'I don\'t have access to real-time', 'GPT Limitation Statement'),
            (r'It\'s important to note that', 'GPT Hedging Pattern')
        ]
        
        for pattern, name in gpt_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                signatures.append({
                    'model': 'GPT',
                    'signature': name,
                    'confidence': 85
                })
        
        # Claude signatures
        claude_patterns = [
            (r'I understand you\'re asking', 'Claude Understanding Pattern'),
            (r'I\'d be happy to help', 'Claude Helpful Pattern'),
            (r'Could you clarify', 'Claude Clarification Pattern')
        ]
        
        for pattern, name in claude_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                signatures.append({
                    'model': 'Claude',
                    'signature': name,
                    'confidence': 80
                })
        
        return signatures
    
    def _calculate_severity(self, count: int, sentence_count: int) -> str:
        """Calculate pattern severity"""
        if sentence_count == 0:
            return 'low'
        
        ratio = count / sentence_count
        if ratio > 0.3:
            return 'high'
        elif ratio > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _get_pattern_description(self, category: str) -> str:
        """Get description for pattern category"""
        descriptions = {
            'formal_transitions': 'Overuse of formal transition words',
            'hedging_phrases': 'Excessive hedging and qualification',
            'ai_favorites': 'Common AI vocabulary choices',
            'conclusion_starters': 'Formulaic conclusion patterns',
            'list_indicators': 'Structured list-like presentation'
        }
        return descriptions.get(category, 'Unknown pattern')
    
    def _calculate_ai_score(self, pattern_scores: Dict[str, str], 
                          pattern_analysis: Dict[str, Any]) -> float:
        """Calculate overall AI score from patterns"""
        score = 50  # Base score
        
        # Add points for pattern severities
        severity_points = {'low': 5, 'medium': 10, 'high': 20}
        for severity in pattern_scores.values():
            score += severity_points.get(severity, 0)
        
        # Add points for consistency
        if pattern_analysis.get('length_consistency', 0) > 0.8:
            score += 15
        
        # Add points for perfect grammar rate
        if pattern_analysis.get('perfect_grammar_rate', 0) > 0.9:
            score += 10
        
        # Low starter diversity suggests AI
        if pattern_analysis.get('starter_diversity', 1) < 0.3:
            score += 10
        
        return min(score, 100)
    
    def _calculate_advanced_ai_score(self, scores: Dict[str, float]) -> float:
        """Calculate advanced AI score"""
        weights = {
            'paragraph': 0.25,
            'repetition': 0.25,
            'formulaic': 0.30,
            'signatures': 0.20
        }
        
        total = sum(
            scores.get(key, 50) * weight 
            for key, weight in weights.items()
        )
        
        return min(total, 100)
