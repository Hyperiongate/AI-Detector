"""
AI & Plagiarism Detection Services Package
"""

# Import all service modules
from .ai_detector import AIDetector
from .text_ai_analyzer import TextAIAnalyzer
from .pattern_analyzer import PatternAnalyzer
from .perplexity_analyzer import PerplexityAnalyzer
from .statistical_analyzer import StatisticalAnalyzer
from .copyleaks_detector import CopyleaksDetector
from .plagiarism_detector import PlagiarismDetector

__all__ = [
    'AIDetector',
    'TextAIAnalyzer',
    'PatternAnalyzer',
    'PerplexityAnalyzer',
    'StatisticalAnalyzer',
    'CopyleaksDetector',
    'PlagiarismDetector'
]
