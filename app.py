"""
AI Detection Analyzer API
Detects AI participation in text and image creation
"""
import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
from typing import Dict, Any, Optional
import base64
from io import BytesIO

# Import AI detection services - FIXED: Using absolute imports
from services.ai_detector import AIDetector
from services.text_ai_analyzer import TextAIAnalyzer
from services.image_ai_analyzer import ImageAIAnalyzer

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize services
try:
    ai_detector = AIDetector()
    text_analyzer = TextAIAnalyzer()
    image_analyzer = ImageAIAnalyzer()
    logger.info("AI detection services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    ai_detector = None
    text_analyzer = None
    image_analyzer = None

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
            'image_analyzer': image_analyzer is not None
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint for AI detection
    Accepts: 
    - Text: { "text": "content to analyze..." }
    - Image: { "image": "base64_encoded_image_data", "image_type": "image/jpeg" }
    - URL: { "url": "https://..." }
    Returns: Comprehensive AI detection analysis
    """
    try:
        # Get request data
        if request.content_type.startswith('multipart/form-data'):
            # Handle file upload
            data = {}
            if 'text' in request.form:
                data['text'] = request.form['text']
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file:
                    # Convert to base64
                    image_data = image_file.read()
                    data['image'] = base64.b64encode(image_data).decode('utf-8')
                    data['image_type'] = image_file.content_type
        else:
            # Handle JSON request
            data = request.get_json()
            
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Determine content type
        text = data.get('text')
        image = data.get('image')
        image_type = data.get('image_type', 'image/jpeg')
        url = data.get('url')
        
        # Check if services are available
        if not ai_detector:
            return jsonify({
                'success': False,
                'error': 'AI detection service is temporarily unavailable'
            }), 503
        
        # Determine if user is pro
        is_pro = data.get('is_pro', False)
        
        # Perform analysis based on content type
        result = None
        
        if text:
            logger.info("Analyzing text content for AI participation")
            result = ai_detector.analyze_text(text, is_pro=is_pro)
            
        elif image:
            logger.info("Analyzing image for AI generation")
            result = ai_detector.analyze_image(image, image_type=image_type, is_pro=is_pro)
            
        elif url:
            logger.info(f"Analyzing content from URL: {url}")
            result = ai_detector.analyze_url(url, is_pro=is_pro)
            
        else:
            return jsonify({
                'success': False,
                'error': 'No content provided. Please provide text, image, or URL.'
            }), 400
        
        # Check if analysis was successful
        if not result.get('success', False):
            error_msg = result.get('error', 'Analysis failed')
            logger.error(f"Analysis failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        logger.info(f"Analysis completed. AI probability: {result.get('ai_probability', 'N/A')}%")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    """
    Extract text from URL for analysis
    Accepts: { "url": "https://..." }
    Returns: Extracted text content
    """
    try:
        data = request.get_json()
        if not data or not data.get('url'):
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        url = data.get('url')
        logger.info(f"Extracting text from: {url}")
        
        if not text_analyzer:
            return jsonify({
                'success': False,
                'error': 'Text extraction service is temporarily unavailable'
            }), 503
        
        # Extract text
        result = text_analyzer.extract_from_url(url)
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Extraction failed')
            logger.error(f"Extraction failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Extraction failed: {str(e)}'
        }), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analysis endpoint for multiple items
    Pro feature only
    """
    try:
        data = request.get_json()
        if not data or not data.get('items'):
            return jsonify({
                'success': False,
                'error': 'Items array is required'
            }), 400
        
        # Check if pro
        if not data.get('is_pro', False):
            return jsonify({
                'success': False,
                'error': 'Batch analysis is a premium feature'
            }), 403
        
        items = data.get('items', [])
        if len(items) > 10:
            return jsonify({
                'success': False,
                'error': 'Maximum 10 items per batch'
            }), 400
        
        # Process batch
        results = []
        for item in items:
            if item.get('text'):
                result = ai_detector.analyze_text(item['text'], is_pro=True)
            elif item.get('image'):
                result = ai_detector.analyze_image(
                    item['image'], 
                    item.get('image_type', 'image/jpeg'), 
                    is_pro=True
                )
            else:
                result = {'success': False, 'error': 'Invalid item type'}
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Batch analysis failed: {str(e)}'
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
        'error': 'File too large. Maximum size is 16MB.'
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
