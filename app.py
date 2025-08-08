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
from werkzeug.utils import secure_filename
import tempfile

# Import AI detection services
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

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        analysis_type = data.get('analysis_type', 'ai')  # 'ai' or 'plagiarism'
        
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
            logger.info(f"Analyzing text content for {analysis_type}")
            
            if analysis_type == 'plagiarism':
                # Mock plagiarism detection for now
                result = {
                    'success': True,
                    'plagiarism_score': 15,
                    'sources_found': 2,
                    'flagged_passages': [
                        {
                            'text': 'The rapid advancement of artificial intelligence',
                            'source_url': 'https://example.com/ai-article',
                            'source_title': 'AI Advances in 2024',
                            'similarity': 85
                        }
                    ],
                    'summary': 'Minor similarities found with common phrases about AI technology.',
                    'analysis_type': 'plagiarism'
                }
            else:
                result = ai_detector.analyze_text(text, is_pro=is_pro)
                result['analysis_type'] = 'ai'
            
        elif image:
            logger.info("Analyzing image for AI generation")
            result = ai_detector.analyze_image(image, image_type=image_type, is_pro=is_pro)
            result['analysis_type'] = 'ai'
            
        elif url:
            logger.info(f"Analyzing content from URL: {url}")
            result = ai_detector.analyze_url(url, is_pro=is_pro)
            result['analysis_type'] = 'ai'
            
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
        
        logger.info(f"Analysis completed. Type: {analysis_type}")
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
    Extract text from uploaded file
    Accepts: multipart/form-data with 'file' field
    Returns: Extracted text content
    """
    try:
        # Check if request has file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: TXT, DOC, DOCX, PDF'
            }), 400
        
        # Get file extension
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        logger.info(f"Extracting text from {file_ext} file: {filename}")
        
        # Extract text based on file type
        text = None
        
        if file_ext == 'txt':
            # Read text file directly
            try:
                text = file.read().decode('utf-8')
            except UnicodeDecodeError:
                # Try different encodings
                file.seek(0)
                try:
                    text = file.read().decode('latin-1')
                except:
                    file.seek(0)
                    text = file.read().decode('utf-8', errors='ignore')
        
        elif file_ext in ['doc', 'docx']:
            # Extract from Word documents
            try:
                import docx2txt
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                    file.save(tmp_file.name)
                    text = docx2txt.process(tmp_file.name)
                    os.unlink(tmp_file.name)
            except ImportError:
                # Fallback: simple extraction for DOCX
                if file_ext == 'docx':
                    try:
                        from zipfile import ZipFile
                        import xml.etree.ElementTree as ET
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                            file.save(tmp_file.name)
                            
                            # Extract text from DOCX
                            text_parts = []
                            with ZipFile(tmp_file.name, 'r') as docx:
                                # Read main document
                                with docx.open('word/document.xml') as xml_file:
                                    tree = ET.parse(xml_file)
                                    root = tree.getroot()
                                    
                                    # Extract all text elements
                                    namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                                    for paragraph in root.iter(namespace + 'p'):
                                        texts = [node.text for node in paragraph.iter(namespace + 't') if node.text]
                                        if texts:
                                            text_parts.append(' '.join(texts))
                            
                            text = '\n'.join(text_parts)
                            os.unlink(tmp_file.name)
                    except Exception as e:
                        logger.error(f"Error extracting from DOCX: {e}")
                        return jsonify({
                            'success': False,
                            'error': 'Failed to extract text from Word document. Please try converting to TXT.'
                        }), 500
                else:
                    return jsonify({
                        'success': False,
                        'error': 'DOC files require additional libraries. Please convert to DOCX or TXT.'
                    }), 500
        
        elif file_ext == 'pdf':
            # Extract from PDF
            try:
                import PyPDF2
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    file.save(tmp_file.name)
                    
                    # Extract text from PDF
                    text_parts = []
                    with open(tmp_file.name, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text_parts.append(page.extract_text())
                    
                    text = '\n'.join(text_parts)
                    os.unlink(tmp_file.name)
            except ImportError:
                # Fallback: basic PDF text extraction
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        file.save(tmp_file.name)
                        
                        # Try to extract readable text
                        with open(tmp_file.name, 'rb') as pdf_file:
                            content = pdf_file.read()
                            # Look for text between BT and ET markers (PDF text objects)
                            import re
                            text_pattern = rb'BT\s*(.*?)\s*ET'
                            matches = re.findall(text_pattern, content, re.DOTALL)
                            
                            text_parts = []
                            for match in matches:
                                # Extract text from PDF commands
                                text_commands = re.findall(rb'\((.*?)\)', match)
                                for cmd in text_commands:
                                    try:
                                        decoded = cmd.decode('utf-8', errors='ignore')
                                        text_parts.append(decoded)
                                    except:
                                        pass
                            
                            text = ' '.join(text_parts)
                            os.unlink(tmp_file.name)
                            
                            if not text.strip():
                                return jsonify({
                                    'success': False,
                                    'error': 'Could not extract text from PDF. The PDF might be scanned or protected.'
                                }), 500
                except Exception as e:
                    logger.error(f"Error extracting from PDF: {e}")
                    return jsonify({
                        'success': False,
                        'error': 'Failed to extract text from PDF. Please try converting to TXT.'
                    }), 500
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text could be extracted from the file'
            }), 400
        
        # Clean up text
        text = text.strip()
        
        # Limit text length for response
        max_length = 50000  # 50k characters max
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': filename,
            'file_type': file_ext,
            'character_count': len(text)
        })
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Text extraction failed: {str(e)}'
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
