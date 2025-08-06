# AI Detection Analyzer

Advanced AI content detection for text and images using multiple detection methods.

## Features

### Free Tier
- **Text Analysis**: Pattern recognition, perplexity analysis, and statistical analysis
- **Image Analysis**: Basic AI artifact detection and metadata analysis
- **Real-time Results**: Instant AI probability scoring
- **Multiple Input Methods**: Direct text, image upload, or URL analysis

### Premium Tier
- **Copyleaks Integration**: Enterprise-grade AI detection with 99.1% accuracy
- **Advanced Pattern Detection**: Deep linguistic analysis and model fingerprinting
- **Image Forensics**: Pixel-level analysis and AI model detection
- **Detailed Reports**: PDF export with comprehensive breakdowns
- **Batch Analysis**: Process multiple items at once

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **AI Detection**: Copyleaks API, Custom algorithms
- **Image Processing**: Pillow, OpenCV
- **Text Analysis**: NLTK, TextStat
- **Deployment**: Render.com

## Setup

### Environment Variables

Create a `.env` file with:

```bash
# Core API Keys
SECRET_KEY=your-secret-key-here
COPYLEAKS_API_KEY=your-copyleaks-api-key
COPYLEAKS_EMAIL=your-copyleaks-email

# Web Scraping (optional)
SCRAPERAPI_KEY=your-scraperapi-key
SCRAPINGBEE_API_KEY=your-scrapingbee-key
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-detection.git
cd ai-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Deployment on Render

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy with these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

## API Endpoints

### `POST /api/analyze`
Main analysis endpoint for AI detection.

**Request Body:**
```json
{
  "text": "Text to analyze",
  // OR
  "image": "base64_encoded_image",
  "image_type": "image/jpeg",
  // OR
  "url": "https://example.com/content",
  
  "is_pro": false  // Premium features flag
}
```

**Response:**
```json
{
  "success": true,
  "ai_probability": 75.5,
  "confidence_level": "High",
  "pattern_analysis": {...},
  "perplexity_analysis": {...},
  "statistical_analysis": {...},
  "summary": "This text shows strong indicators of AI generation..."
}
```

### `GET /api/health`
Health check endpoint.

## Detection Methods

### Text Detection

1. **Pattern Analysis**
   - AI-specific phrases and transitions
   - Sentence structure patterns
   - Formulaic writing detection

2. **Perplexity Analysis**
   - Text predictability measurement
   - Burstiness (sentence variation)
   - Vocabulary richness

3. **Statistical Analysis**
   - Word length consistency
   - Sentence length variance
   - Punctuation patterns

4. **Copyleaks Integration** (Premium)
   - Professional AI detection API
   - 99.1% accuracy rate
   - Section-by-section analysis

### Image Detection

1. **Metadata Analysis**
   - EXIF data inspection
   - AI tool signatures

2. **Pixel Analysis**
   - Color distribution
   - Gradient perfection
   - Symmetry detection

3. **Artifact Detection**
   - Grid patterns
   - Unnatural edges
   - Texture anomalies

4. **Forensic Analysis** (Premium)
   - Error level analysis
   - Compression artifacts
   - Model fingerprinting

## Project Structure

```
ai-detection/
├── app.py                 # Main Flask application
├── services/
│   ├── ai_detector.py     # Main orchestrator
│   ├── text_ai_analyzer.py
│   ├── image_ai_analyzer.py
│   ├── pattern_analyzer.py
│   ├── perplexity_analyzer.py
│   ├── statistical_analyzer.py
│   └── copyleaks_detector.py
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── app.js
│       └── components.js
├── templates/
│   └── index.html
├── requirements.txt
├── .env.example
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues or questions, please open a
