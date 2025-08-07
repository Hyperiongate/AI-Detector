"""
AI & Plagiarism Detection API
Detects AI-generated content and checks for plagiarism in text
"""
import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
from typing import Dict, Any, Optional
import base64
from io import BytesIO

# Import detection services
from services.ai_detector import AIDetector
from services.text_ai_analyzer import TextAIAnalyzer
from services.plagiarism_detector import PlagiarismDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with explicit static folder configuration
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max for text

# Initialize services
try:
    ai_detector = AIDetector()
    text_analyzer = TextAIAnalyzer()
    plagiarism_detector = PlagiarismDetector()
    logger.info("Detection services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    ai_detector = None
    text_analyzer = None
    plagiarism_detector = None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'ai_detector': ai_detector is not None,
            'text_analyzer': text_analyzer is not None,
            'plagiarism_detector': plagiarism_detector is not None
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint for AI detection and plagiarism checking
    Accepts: 
    - Text: { "text": "content to analyze...", "analysis_type": "ai" or "plagiarism" }
    Returns: Comprehensive analysis results
    """
    try:
        # Get request data
        data = request.get_json()
            
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Get parameters
        text = data.get('text')
        analysis_type = data.get('analysis_type', 'ai')
        is_pro = data.get('is_pro', False)
        
        # Validate input
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
            
        if len(text) < 50:
            return jsonify({
                'success': False,
                'error': 'Text must be at least 50 characters long'
            }), 400
            
        if len(text) > 50000:
            return jsonify({
                'success': False,
                'error': 'Text must be less than 50,000 characters'
            }), 400
        
        # Check if services are available
        if analysis_type == 'ai' and not ai_detector:
            return jsonify({
                'success': False,
                'error': 'AI detection service is temporarily unavailable'
            }), 503
            
        if analysis_type == 'plagiarism' and not plagiarism_detector:
            return jsonify({
                'success': False,
                'error': 'Plagiarism detection service is temporarily unavailable'
            }), 503
        
        # Perform analysis
        result = None
        
        if analysis_type == 'plagiarism':
            logger.info("Performing plagiarism check")
            result = plagiarism_detector.check_plagiarism(text, is_pro=is_pro)
        else:  # Default to AI detection
            logger.info("Performing AI detection analysis")
            result = ai_detector.analyze_text(text, is_pro=is_pro)
        
        # Check if analysis was successful
        if not result.get('success', False):
            error_msg = result.get('error', 'Analysis failed')
            logger.error(f"Analysis failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        logger.info(f"Analysis completed. Type: {analysis_type}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    """
    Generate PDF report for analysis results
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        analysis_type = data.get('analysis_type', 'ai')
        
        # For now, return a simple message
        # In production, you would use a PDF library like ReportLab
        return jsonify({
            'success': False,
            'error': 'PDF generation is not yet implemented'
        }), 501
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'PDF generation failed: {str(e)}'
        }), 500

# Serve static files explicitly
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'success': False,
        'error': 'Content too large. Maximum size is 5MB.'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Get port from environment variable (for deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
