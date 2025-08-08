// static/js/app.js - AI & Plagiarism Detection Application Logic

class TextAnalysisApp {
    constructor() {
        this.currentAnalysis = null;
        this.isPremium = false;
        this.analysisType = 'ai'; // 'ai' or 'plagiarism'
        this.API_ENDPOINT = '/api/analyze';
        this.progressInterval = null;
        this.analysisStartTime = null;
        this.uploadedFile = null;
        this.extractedText = '';
        
        this.init();
    }

    init() {
        // Set up event listeners
        this.setupEventListeners();
        
        // Check for demo mode
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('demo')) {
            this.loadDemoContent();
        }
    }

    setupEventListeners() {
        // Analysis type selection
        document.querySelectorAll('input[name="analysisType"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.analysisType = e.target.value;
                this.updateAnalyzeButton();
            });
        });

        // File upload drag and drop
        const uploadArea = document.getElementById('fileUploadArea');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileSelection(files[0]);
                }
            });
        }

        // Global functions
        window.analyzeContent = () => this.analyzeContent();
        window.unlockPremium = () => this.unlockPremium();
        window.downloadPDF = () => this.downloadPDF();
        window.shareAnalysis = () => this.shareAnalysis();
        window.showPricing = () => this.showPricing();
        window.showHowItWorks = () => this.showHowItWorks();
        window.hideHowItWorks = () => this.hideHowItWorks();
        window.updateCharCount = () => this.updateCharCount();
        window.clearText = () => this.clearText();
        window.switchInputTab = (tab) => this.switchInputTab(tab);
        window.handleFileUpload = (event) => this.handleFileUpload(event);
        window.removeFile = () => this.removeFile();
    }

    switchInputTab(tab) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.style.display = 'none';
        });
        
        // Remove active class from all tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab content
        document.getElementById(`${tab}-tab`).style.display = 'block';
        
        // Add active class to selected tab button
        document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
        
        // Update analyze button based on content availability
        if (tab === 'file') {
            const fileAnalyzeBtn = document.getElementById('fileAnalyzeBtn');
            fileAnalyzeBtn.disabled = !this.extractedText;
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFileSelection(file);
        }
    }

    handleFileSelection(file) {
        // Validate file type
        const validTypes = ['.txt', '.doc', '.docx', '.pdf'];
        const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        
        if (!validTypes.includes(fileExtension)) {
            this.showError('Please upload a valid file type (TXT, DOC, DOCX, or PDF)');
            return;
        }

        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
            this.showError('File size must be less than 5MB');
            return;
        }

        this.uploadedFile = file;
        
        // Show file preview
        document.getElementById('filePreview').style.display = 'block';
        document.querySelector('.upload-label').style.display = 'none';
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
        
        // Extract text based on file type
        if (fileExtension === '.txt') {
            this.extractTextFromTxtFile(file);
        } else {
            // For DOC, DOCX, PDF - would need server-side processing
            this.extractTextFromDocument(file);
        }
    }

    extractTextFromTxtFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.extractedText = e.target.result;
            this.showExtractedTextPreview();
        };
        reader.onerror = () => {
            this.showError('Failed to read file');
        };
        reader.readAsText(file);
    }

    extractTextFromDocument(file) {
        // Show loading state
        document.getElementById('extractedTextPreview').style.display = 'block';
        document.getElementById('textPreview').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Extracting text...';
        document.getElementById('fileCharCount').textContent = '';
        
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Call API to extract text
        fetch('/api/extract-text', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to extract text');
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.text) {
                this.extractedText = data.text;
                this.showExtractedTextPreview();
            } else {
                throw new Error(data.error || 'Failed to extract text');
            }
        })
        .catch(error => {
            console.error('Text extraction error:', error);
            this.showError('Failed to extract text from document. Please try a TXT file.');
            document.getElementById('extractedTextPreview').style.display = 'none';
        });
    }

    showExtractedTextPreview() {
        const preview = document.getElementById('extractedTextPreview');
        const textPreview = document.getElementById('textPreview');
        const charCount = document.getElementById('fileCharCount');
        
        preview.style.display = 'block';
        
        // Show first 500 characters as preview
        const previewText = this.extractedText.substring(0, 500);
        textPreview.textContent = previewText + (this.extractedText.length > 500 ? '...' : '');
        
        // Update character count
        charCount.textContent = `${this.extractedText.length} characters`;
        
        // Enable analyze button
        document.getElementById('fileAnalyzeBtn').disabled = false;
        
        // Update analyze button text
        const btn = document.getElementById('fileAnalyzeBtn');
        if (this.analysisType === 'plagiarism') {
            btn.innerHTML = '<i class="fas fa-search"></i> <span>Check for Plagiarism</span>';
        } else {
            btn.innerHTML = '<i class="fas fa-search"></i> <span>Check for AI</span>';
        }
    }

    removeFile() {
        this.uploadedFile = null;
        this.extractedText = '';
        
        // Reset UI
        document.getElementById('filePreview').style.display = 'none';
        document.querySelector('.upload-label').style.display = 'flex';
        document.getElementById('fileInput').value = '';
        document.getElementById('fileAnalyzeBtn').disabled = true;
        
        // Clear file info
        document.getElementById('fileName').textContent = '';
        document.getElementById('fileSize').textContent = '';
        document.getElementById('textPreview').textContent = '';
        document.getElementById('fileCharCount').textContent = '';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    updateAnalyzeButton() {
        const textBtn = document.getElementById('analyzeBtn');
        const fileBtn = document.getElementById('fileAnalyzeBtn');
        
        const buttonText = this.analysisType === 'plagiarism' 
            ? '<i class="fas fa-search"></i> <span>Check for Plagiarism</span>'
            : '<i class="fas fa-search"></i> <span>Check for AI</span>';
        
        textBtn.innerHTML = buttonText;
        fileBtn.innerHTML = buttonText;
    }

    updateCharCount() {
        const text = document.getElementById('textInput').value;
        document.getElementById('charCount').textContent = `${text.length} characters`;
    }

    clearText() {
        document.getElementById('textInput').value = '';
        this.updateCharCount();
    }

    showHowItWorks() {
        document.getElementById('howItWorksSection').style.display = 'flex';
    }

    hideHowItWorks() {
        document.getElementById('howItWorksSection').style.display = 'none';
    }

    async analyzeContent() {
        this.analysisStartTime = Date.now();
        
        // Get text based on active tab
        let text = '';
        const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
        
        if (activeTab === 'text') {
            text = document.getElementById('textInput').value.trim();
        } else if (activeTab === 'file') {
            text = this.extractedText;
        }
        
        if (!text || text.length < 50) {
            this.showError('Please enter at least 50 characters of text');
            return;
        }

        // Prepare request data
        const requestData = {
            text: text,
            analysis_type: this.analysisType,
            is_pro: this.isPremium
        };

        // Reset UI
        this.hideError();
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('aiResults').style.display = 'none';
        document.getElementById('plagiarismResults').style.display = 'none';
        document.getElementById('premiumAnalysis').style.display = 'none';
        document.getElementById('premiumCTA').style.display = 'block';
        
        // Show progress
        this.showProgress();
        
        // Disable analyze buttons
        const textAnalyzeBtn = document.getElementById('analyzeBtn');
        const fileAnalyzeBtn = document.getElementById('fileAnalyzeBtn');
        textAnalyzeBtn.disabled = true;
        fileAnalyzeBtn.disabled = true;
        textAnalyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        fileAnalyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        try {
            // Call API
            const response = await fetch(this.API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Analysis failed');
            }

            const data = await response.json();
            this.currentAnalysis = data;
            
            // Hide progress
            this.hideProgress();
            
            // Display results
            this.displayResults(data);
            
            // Log success
            console.log('Analysis complete:', data);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'An error occurred during analysis');
            this.hideProgress();
        } finally {
            // Re-enable buttons
            textAnalyzeBtn.disabled = false;
            fileAnalyzeBtn.disabled = activeTab === 'file' && !this.extractedText;
            this.updateAnalyzeButton();
        }
    }

    showProgress() {
        const progressSection = document.getElementById('progressSection');
        progressSection.style.display = 'block';
        
        // Update title and show correct stages
        const title = document.getElementById('progressTitle');
        const aiStages = document.getElementById('aiProgressStages');
        const plagiarismStages = document.getElementById('plagiarismProgressStages');
        
        if (this.analysisType === 'plagiarism') {
            title.textContent = 'Plagiarism Check in Progress';
            aiStages.style.display = 'none';
            plagiarismStages.style.display = 'flex';
            this.animateProgressStages(['submit', 'internet', 'academic', 'similarity', 'report']);
        } else {
            title.textContent = 'AI Detection Analysis in Progress';
            aiStages.style.display = 'flex';
            plagiarismStages.style.display = 'none';
            this.animateProgressStages(['extract', 'patterns', 'perplexity', 'statistical', 'score']);
        }
    }

    animateProgressStages(stages) {
        let currentStage = 0;
        
        // Reset progress
        document.getElementById('progressFill').style.width = '0%';
        document.querySelectorAll('.stage').forEach(stage => {
            stage.classList.remove('active', 'complete');
        });
        
        // Animate stages
        this.progressInterval = setInterval(() => {
            if (currentStage < stages.length) {
                // Mark current stage as active
                const stageElements = document.querySelectorAll(
                    `#${this.analysisType === 'plagiarism' ? 'plagiarismProgressStages' : 'aiProgressStages'} .stage`
                );
                
                stageElements.forEach((stage, index) => {
                    if (index < currentStage) {
                        stage.classList.add('complete');
                        stage.classList.remove('active');
                    } else if (index === currentStage) {
                        stage.classList.add('active');
                    }
                });
                
                // Update progress bar
                const progress = ((currentStage + 1) / stages.length) * 100;
                document.getElementById('progressFill').style.width = `${progress}%`;
                
                // Update message
                document.getElementById('progressMessage').textContent = 
                    this.getProgressMessage(stages[currentStage]);
                
                currentStage++;
            }
        }, 800);
    }

    getProgressMessage(stage) {
        const messages = {
            // AI stages
            'extract': 'Processing your text...',
            'patterns': 'Analyzing AI writing patterns...',
            'perplexity': 'Calculating text perplexity...',
            'statistical': 'Running statistical analysis...',
            'score': 'Calculating final AI probability...',
            // Plagiarism stages
            'submit': 'Submitting text for analysis...',
            'internet': 'Searching billions of web pages...',
            'academic': 'Checking academic databases...',
            'similarity': 'Analyzing text similarity...',
            'report': 'Generating plagiarism report...'
        };
        return messages[stage] || 'Processing...';
    }

    hideProgress() {
        const progressSection = document.getElementById('progressSection');
        progressSection.style.display = 'none';
        
        // Clear interval
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    displayResults(data) {
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Calculate analysis time
        const analysisTime = ((Date.now() - this.analysisStartTime) / 1000).toFixed(1);
        
        // Smooth scroll to results
        setTimeout(() => {
            document.getElementById('resultsSection').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 100);

        if (this.analysisType === 'plagiarism') {
            this.displayPlagiarismResults(data, analysisTime);
        } else {
            this.displayAIResults(data, analysisTime);
        }
    }

    displayAIResults(data, analysisTime) {
        // Show AI results container
        document.getElementById('aiResults').style.display = 'block';
        document.getElementById('plagiarismResults').style.display = 'none';
        
        // Display AI probability with animation
        const aiProbability = data.ai_probability || 50;
        this.animateAIProbability(aiProbability);
        
        // Display summary
        document.getElementById('aiSummaryText').textContent = data.summary || 'Analysis complete';
        
        // Update quick stats
        const patternCount = data.pattern_analysis?.detected_patterns?.length || 0;
        document.getElementById('patternCount').textContent = patternCount;
        document.getElementById('methodsUsed').textContent = '5+';
        document.getElementById('confidenceRate').textContent = Math.round(this.calculateConfidence(data));
        
        // Display basic findings (free tier)
        this.displayBasicAIFindings(data);
        
        // If premium, show all analysis
        if (this.isPremium && data.is_pro) {
            this.displayPremiumAIAnalysis(data);
        }
    }

    displayPlagiarismResults(data, analysisTime) {
        // Show plagiarism results container
        document.getElementById('plagiarismResults').style.display = 'block';
        document.getElementById('aiResults').style.display = 'none';
        
        // Display plagiarism score
        const plagiarismScore = data.plagiarism_score || 0;
        this.animatePlagiarismScore(plagiarismScore);
        
        // Display summary
        document.getElementById('plagiarismSummaryText').textContent = data.summary || 'Analysis complete';
        
        // Update quick stats
        document.getElementById('similarityPercent').textContent = plagiarismScore;
        document.getElementById('sourcesFound').textContent = data.sources_found || 0;
        document.getElementById('flaggedPassages').textContent = data.flagged_passages?.length || 0;
        
        // Display plagiarized sections if any
        if (data.flagged_passages && data.flagged_passages.length > 0) {
            this.displayPlagiarizedSections(data.flagged_passages);
        }
        
        // If premium, show detailed analysis
        if (this.isPremium && data.is_pro) {
            this.displayPremiumPlagiarismAnalysis(data);
        }
    }

    animateAIProbability(probability) {
        // Create gauge chart
        this.createAIProbabilityGauge('aiProbabilityGauge', probability);
        
        // Set label and description
        const label = document.getElementById('aiLabel');
        const description = document.getElementById('aiDescription');
        
        let labelText, descText;
        if (probability >= 90) {
            labelText = 'AI Generated';
            descText = 'This content is almost certainly AI-generated with very high confidence.';
        } else if (probability >= 70) {
            labelText = 'Likely AI Generated';
            descText = 'Strong indicators suggest this content was created by AI.';
        } else if (probability >= 50) {
            labelText = 'Possibly AI Generated';
            descText = 'Mixed signals indicate possible AI involvement or heavy AI editing.';
        } else if (probability >= 30) {
            labelText = 'Likely Human';
            descText = 'Most indicators suggest human authorship with possible AI assistance.';
        } else {
            labelText = 'Human Created';
            descText = 'This content appears to be created by a human with minimal AI involvement.';
        }
        
        // Animate text appearance
        setTimeout(() => {
            label.textContent = labelText;
            label.style.opacity = '0';
            label.style.animation = 'fadeIn 0.5s forwards';
        }, 500);
        
        setTimeout(() => {
            description.textContent = descText;
            description.style.opacity = '0';
            description.style.animation = 'fadeIn 0.5s forwards';
        }, 700);
    }

    animatePlagiarismScore(score) {
        // Create gauge chart
        this.createPlagiarismGauge('plagiarismGauge', score);
        
        // Set label and description
        const label = document.getElementById('plagiarismLabel');
        const description = document.getElementById('plagiarismDescription');
        
        let labelText, descText;
        if (score >= 80) {
            labelText = 'High Plagiarism Detected';
            descText = 'Significant portions of this text match existing sources.';
        } else if (score >= 50) {
            labelText = 'Moderate Plagiarism';
            descText = 'Several passages appear to be copied or closely paraphrased.';
        } else if (score >= 20) {
            labelText = 'Some Similarity Found';
            descText = 'Minor similarities detected, possibly common phrases or citations.';
        } else if (score >= 10) {
            labelText = 'Minimal Similarity';
            descText = 'Very few matches found, likely original content.';
        } else {
            labelText = 'Original Content';
            descText = 'This appears to be completely original content.';
        }
        
        // Animate text appearance
        setTimeout(() => {
            label.textContent = labelText;
            label.style.opacity = '0';
            label.style.animation = 'fadeIn 0.5s forwards';
        }, 500);
        
        setTimeout(() => {
            description.textContent = descText;
            description.style.opacity = '0';
            description.style.animation = 'fadeIn 0.5s forwards';
        }, 700);
    }

    createAIProbabilityGauge(canvasId, probability) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Settings
        const centerX = canvas.width / 2;
        const centerY = canvas.height - 20;
        const radius = 80;
        const startAngle = Math.PI;
        const endAngle = 2 * Math.PI;
        
        // Draw background arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.lineWidth = 20;
        ctx.strokeStyle = '#e5e7eb';
        ctx.stroke();
        
        // Draw probability arc
        const probAngle = startAngle + (probability / 100) * Math.PI;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, probAngle);
        ctx.lineWidth = 20;
        
        // Color based on probability
        if (probability >= 70) {
            ctx.strokeStyle = '#ef4444'; // Red for high AI probability
        } else if (probability >= 50) {
            ctx.strokeStyle = '#f59e0b'; // Orange for medium
        } else if (probability >= 30) {
            ctx.strokeStyle = '#3b82f6'; // Blue for low
        } else {
            ctx.strokeStyle = '#10b981'; // Green for very low
        }
        
        ctx.stroke();
        
        // Draw percentage text
        ctx.font = 'bold 36px Arial';
        ctx.fillStyle = '#1f2937';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(probability)}%`, centerX, centerY - 10);
        
        // Draw label
        ctx.font = '14px Arial';
        ctx.fillStyle = '#6b7280';
        ctx.fillText('AI Probability', centerX, centerY + 15);
    }

    createPlagiarismGauge(canvasId, score) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Settings
        const centerX = canvas.width / 2;
        const centerY = canvas.height - 20;
        const radius = 80;
        const startAngle = Math.PI;
        const endAngle = 2 * Math.PI;
        
        // Draw background arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.lineWidth = 20;
        ctx.strokeStyle = '#e5e7eb';
        ctx.stroke();
        
        // Draw score arc
        const scoreAngle = startAngle + (score / 100) * Math.PI;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, scoreAngle);
        ctx.lineWidth = 20;
        
        // Color based on score
        if (score >= 50) {
            ctx.strokeStyle = '#ef4444'; // Red for high plagiarism
        } else if (score >= 20) {
            ctx.strokeStyle = '#f59e0b'; // Orange for moderate
        } else if (score >= 10) {
            ctx.strokeStyle = '#3b82f6'; // Blue for low
        } else {
            ctx.strokeStyle = '#10b981'; // Green for original
        }
        
        ctx.stroke();
        
        // Draw percentage text
        ctx.font = 'bold 36px Arial';
        ctx.fillStyle = '#1f2937';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(score)}%`, centerX, centerY - 10);
        
        // Draw label
        ctx.font = '14px Arial';
        ctx.fillStyle = '#6b7280';
        ctx.fillText('Similarity Score', centerX, centerY + 15);
    }

    displayBasicAIFindings(data) {
        const findingsEl = document.getElementById('basicAIFindings');
        let html = '';
        
        // AI probability summary
        const aiProb = data.ai_probability || 50;
        if (aiProb >= 70) {
            html += '<p><strong>High AI probability detected.</strong> This text shows multiple strong indicators of AI generation.</p>';
        } else if (aiProb >= 50) {
            html += '<p><strong>Moderate AI probability.</strong> Several AI patterns were found, suggesting possible AI involvement.</p>';
        } else {
            html += '<p><strong>Low AI probability.</strong> This content appears to be predominantly human-written.</p>';
        }
        
        html += '<ul>';
        
        // Pattern findings
        if (data.pattern_analysis?.detected_patterns?.length > 0) {
            const patternCount = data.pattern_analysis.detected_patterns.length;
            html += `<li><i class="fas fa-brain text-info"></i> <strong>${patternCount} AI writing patterns detected</strong>, including formal transitions and common AI phrases.</li>`;
        }
        
        // Perplexity findings
        if (data.perplexity_analysis) {
            const perplexity = data.perplexity_analysis.perplexity || 0;
            if (perplexity < 50) {
                html += '<li><i class="fas fa-chart-line text-warning"></i> <strong>Low text perplexity</strong> indicates unusually predictable text structure.</li>';
            }
        }
        
        // Statistical findings
        if (data.statistical_analysis?.vocabulary_diversity) {
            const diversity = data.statistical_analysis.vocabulary_diversity;
            if (diversity < 0.3) {
                html += '<li><i class="fas fa-calculator text-warning"></i> <strong>Limited vocabulary diversity</strong> with repetitive word usage patterns.</li>';
            }
        }
        
        // General advice
        if (aiProb >= 50) {
            html += '<li><i class="fas fa-lightbulb text-primary"></i> <strong>Recommendation:</strong> Review the text for AI-generated sections and consider rewriting for authenticity.</li>';
        } else {
            html += '<li><i class="fas fa-check-circle text-success"></i> <strong>Good news:</strong> Your text appears largely original with natural human writing characteristics.</li>';
        }
        
        html += '</ul>';
        
        findingsEl.innerHTML = html;
    }

    displayPlagiarizedSections(flaggedPassages) {
        const detailsCard = document.getElementById('plagiarismDetailsCard');
        const sectionsEl = document.getElementById('plagiarizedSections');
        
        detailsCard.style.display = 'block';
        
        let html = '';
        flaggedPassages.forEach((passage, index) => {
            html += `
                <div class="plagiarized-item">
                    <div class="plagiarized-text">
                        "${passage.text}"
                    </div>
                    <div class="plagiarized-source">
                        <i class="fas fa-link"></i> 
                        <strong>Source:</strong> 
                        <a href="${passage.source_url}" target="_blank">${passage.source_title || passage.source_url}</a>
                        <span class="similarity-badge">${passage.similarity}% match</span>
                    </div>
                </div>
            `;
        });
        
        sectionsEl.innerHTML = html;
    }

    calculateConfidence(data) {
        // Calculate confidence based on multiple factors
        let confidence = 60; // Base confidence
        
        // Add confidence for each analysis method
        if (data.pattern_analysis) confidence += 10;
        if (data.perplexity_analysis) confidence += 10;
        if (data.statistical_analysis) confidence += 10;
        if (data.copyleaks_analysis) confidence += 15;
        
        // Adjust based on agreement between methods
        const scores = [];
        if (data.pattern_analysis?.ai_patterns_score) scores.push(data.pattern_analysis.ai_patterns_score);
        if (data.perplexity_analysis?.ai_probability) scores.push(data.perplexity_analysis.ai_probability);
        if (data.statistical_analysis?.ai_probability) scores.push(data.statistical_analysis.ai_probability);
        
        if (scores.length > 1) {
            const variance = this.calculateVariance(scores);
            if (variance < 10) confidence += 10; // High agreement
            else if (variance > 30) confidence -= 10; // Low agreement
        }
        
        return Math.min(confidence, 95);
    }

    calculateVariance(scores) {
        const mean = scores.reduce((a, b) => a + b) / scores.length;
        const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
        return Math.sqrt(variance);
    }

    unlockPremium() {
        if (!this.currentAnalysis) return;
        
        // In a real app, this would check payment/subscription
        this.isPremium = true;
        
        // Hide CTA
        document.getElementById('premiumCTA').style.display = 'none';
        
        // Re-analyze with premium features
        this.analyzeContent();
    }

    displayPremiumAIAnalysis(data) {
        const premiumSection = document.getElementById('premiumAnalysis');
        premiumSection.style.display = 'block';
        
        // Clear previous content
        const grid = document.getElementById('analysisGrid');
        grid.innerHTML = '';
        
        // Create all analysis cards
        const cards = [];
        
        // Pattern Analysis Card
        if (data.pattern_analysis || data.advanced_patterns) {
            cards.push(this.createPatternAnalysisCard(data));
        }
        
        // Perplexity Analysis Card
        if (data.perplexity_analysis) {
            cards.push(this.createPerplexityCard(data));
        }
        
        // Statistical Analysis Card
        if (data.statistical_analysis || data.linguistic_analysis) {
            cards.push(this.createStatisticalCard(data));
        }
        
        // Copyleaks Analysis Card
        if (data.copyleaks_analysis) {
            cards.push(this.createCopyleaksCard(data));
        }
        
        // Model Detection Card
        if (data.model_detection) {
            cards.push(this.createModelDetectionCard(data));
        }
        
        // Section-by-Section Analysis
        if (data.section_analysis) {
            cards.push(this.createSectionAnalysisCard(data));
        }
        
        // Add all cards with staggered animation
        cards.forEach((cardHtml, index) => {
            setTimeout(() => {
                grid.innerHTML += cardHtml;
            }, index * 100);
        });
        
        // Smooth scroll to premium section
        setTimeout(() => {
            premiumSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
    }

    displayPremiumPlagiarismAnalysis(data) {
        const premiumSection = document.getElementById('premiumAnalysis');
        premiumSection.style.display = 'block';
        
        // Clear previous content
        const grid = document.getElementById('analysisGrid');
        grid.innerHTML = '';
        
        // Create plagiarism-specific cards
        const cards = [];
        
        // Source breakdown
        if (data.source_breakdown) {
            cards.push(this.createSourceBreakdownCard(data));
        }
        
        // Detailed matches
        if (data.detailed_matches) {
            cards.push(this.createDetailedMatchesCard(data));
        }
        
        // Paraphrase detection
        if (data.paraphrase_detection) {
            cards.push(this.createParaphraseCard(data));
        }
        
        // Add all cards
        cards.forEach((cardHtml, index) => {
            setTimeout(() => {
                grid.innerHTML += cardHtml;
            }, index * 100);
        });
    }

    createPatternAnalysisCard(data) {
        const patterns = data.pattern_analysis?.detected_patterns || [];
        const score = data.pattern_analysis?.ai_patterns_score || 0;
        
        let html = `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-brain"></i> Pattern Analysis</h3>
                    <span class="card-score ${this.getScoreClass(score)}">${score}%</span>
                </div>
                <div class="card-content">
        `;
        
        if (patterns.length > 0) {
            html += '<div class="pattern-list">';
            patterns.forEach(pattern => {
                html += `
                    <div class="pattern-item severity-${pattern.severity}">
                        <div>
                            <span class="pattern-name">${pattern.type}</span>
                            <div class="pattern-description">${pattern.description}</div>
                        </div>
                        <span class="pattern-count">${pattern.count} found</span>
                    </div>
                `;
            });
            html += '</div>';
        }
        
        // Advanced patterns (pro)
        if (data.advanced_patterns) {
            html += `
                <div class="ai-insight">
                    <h5>Advanced Pattern Detection</h5>
                    <p>Formulaic structure score: ${data.advanced_patterns.formulaic_score}%</p>
                    <p>Paragraph consistency: ${data.advanced_patterns.paragraph_analysis?.consistency_score || 'N/A'}</p>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }

    createPerplexityCard(data) {
        const perplexity = data.perplexity_analysis?.perplexity || 0;
        const burstiness = data.perplexity_analysis?.burstiness || 0;
        const aiProb = data.perplexity_analysis?.ai_probability || 0;
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-chart-area"></i> Perplexity Analysis</h3>
                    <span class="card-score ${this.getScoreClass(aiProb)}">${aiProb}%</span>
                </div>
                <div class="card-content">
                    <div class="metric-grid">
                        <div class="metric-item">
                            <label>Perplexity Score</label>
                            <span class="metric-value">${perplexity.toFixed(2)}</span>
                            <small>${perplexity < 50 ? 'Low (AI-like)' : 'High (Human-like)'}</small>
                        </div>
                        <div class="metric-item">
                            <label>Burstiness</label>
                            <span class="metric-value">${burstiness.toFixed(2)}</span>
                            <small>${burstiness < 0.5 ? 'Low variation' : 'High variation'}</small>
                        </div>
                    </div>
                    <div class="ai-insight">
                        <p>${this.getPerplexityInsight(perplexity, burstiness)}</p>
                    </div>
                </div>
            </div>
        `;
    }

    createStatisticalCard(data) {
        const stats = data.statistical_analysis || {};
        const linguistic = data.linguistic_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-calculator"></i> Statistical Analysis</h3>
                    <span class="card-score ${this.getScoreClass(stats.ai_probability || 0)}">${stats.ai_probability || 0}%</span>
                </div>
                <div class="card-content">
                    <div class="stat-grid">
                        <div class="stat-item">
                            <label>Vocabulary Diversity</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${(stats.vocabulary_diversity || 0) * 100}%"></div>
                            </div>
                        </div>
                        <div class="stat-item">
                            <label>Sentence Variance</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${Math.min(stats.sentence_variance || 0, 100)}%"></div>
                            </div>
                        </div>
                    </div>
                    ${linguistic.coherence_score ? `
                        <div class="linguistic-metrics">
                            <h5>Linguistic Analysis</h5>
                            <p>Coherence: ${linguistic.coherence_score}%</p>
                            <p>Style Consistency: ${linguistic.style_consistency}%</p>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    createCopyleaksCard(data) {
        const copyleaks = data.copyleaks_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-shield-alt"></i> Copyleaks AI Detection</h3>
                    <span class="card-score ${this.getScoreClass(copyleaks.ai_probability || 0)}">${copyleaks.ai_probability || 0}%</span>
                </div>
                <div class="card-content">
                    <div class="copyleaks-result">
                        <div class="ai-confidence-badge ${this.getConfidenceClass(copyleaks.ai_probability)}">
                            <i class="fas fa-certificate"></i>
                            <span>Enterprise-Grade Detection</span>
                        </div>
                        <p class="detection-summary">${copyleaks.summary || 'Professional AI detection analysis completed.'}</p>
                        ${copyleaks.detected_sections ? `
                            <div class="detected-sections">
                                <h5>AI-Generated Sections Detected:</h5>
                                <ul>
                                    ${copyleaks.detected_sections.map(section => 
                                        `<li>${section}</li>`
                                    ).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    createModelDetectionCard(data) {
        const detection = data.model_detection || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-fingerprint"></i> AI Model Detection</h3>
                    <span class="card-score">${detection.confidence || 0}%</span>
                </div>
                <div class="card-content">
                    <div class="model-detection">
                        <div class="detected-model">
                            <i class="fas fa-robot"></i>
                            <h4>${detection.detected_model || 'Unknown Model'}</h4>
                        </div>
                        ${detection.model_signatures && detection.model_signatures.length > 0 ? `
                            <div class="model-signatures">
                                <h5>Model Signatures Found:</h5>
                                ${detection.model_signatures.map(sig => `
                                    <div class="signature-item">
                                        <span class="model-name">${sig.model}</span>
                                        <span class="signature-type">${sig.signature}</span>
                                        <span class="confidence">${sig.confidence}% match</span>
                                    </div>
                                `).join('')}
                            </div>
                        ` : '<p>No specific model signatures detected.</p>'}
                    </div>
                </div>
            </div>
        `;
    }

    createSectionAnalysisCard(data) {
        const sections = data.section_analysis || [];
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-align-left"></i> Section-by-Section Analysis</h3>
                </div>
                <div class="card-content">
                    <div class="section-analysis">
                        ${sections.map((section, index) => `
                            <div class="section-item">
                                <h5>Section ${index + 1}</h5>
                                <p class="section-text">"${section.text.substring(0, 100)}..."</p>
                                <div class="section-score">
                                    AI Probability: <span class="${this.getScoreClass(section.ai_probability)}">${section.ai_probability}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    getScoreClass(score) {
        if (score >= 70) return 'score-high';
        if (score >= 50) return 'score-medium';
        if (score >= 30) return 'score-low';
        return 'score-very-low';
    }

    getConfidenceClass(probability) {
        if (probability >= 70) return 'high';
        if (probability >= 40) return 'medium';
        return 'low';
    }

    getPerplexityInsight(perplexity, burstiness) {
        if (perplexity < 30 && burstiness < 0.3) {
            return "Very low perplexity and burstiness strongly indicate AI generation. The text is highly predictable with minimal variation.";
        } else if (perplexity < 50) {
            return "Low perplexity suggests AI involvement. The text follows predictable patterns common in AI-generated content.";
        } else if (perplexity > 100 && burstiness > 0.7) {
            return "High perplexity and burstiness indicate human writing. The text shows natural variation and unpredictability.";
        } else {
            return "Mixed perplexity signals. The text may involve both human and AI contributions.";
        }
    }

    async downloadPDF() {
        if (!this.currentAnalysis) return;
        
        // Show loading overlay
        document.getElementById('loadingOverlay').style.display = 'flex';
        
        try {
            const response = await fetch('/api/generate-pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...this.currentAnalysis,
                    analysis_type: this.analysisType
                })
            });
            
            if (!response.ok) throw new Error('PDF generation failed');
            
            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.analysisType}-detection-report-${Date.now()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
        } catch (error) {
            console.error('PDF download error:', error);
            this.showError('Failed to generate PDF. Please try again.');
        } finally {
            document.getElementById('loadingOverlay').style.display = 'none';
        }
    }

    shareAnalysis() {
        if (!this.currentAnalysis) return;
        
        const score = this.analysisType === 'plagiarism' 
            ? this.currentAnalysis.plagiarism_score 
            : this.currentAnalysis.ai_probability;
            
        const text = this.analysisType === 'plagiarism'
            ? `Plagiarism Check Result: ${score}% similarity found.`
            : `AI Detection Result: ${score}% probability of AI generation.`;
        
        if (navigator.share) {
            navigator.share({
                title: `${this.analysisType === 'plagiarism' ? 'Plagiarism' : 'AI'} Detection Analysis`,
                text: text,
                url: window.location.href
            }).catch(err => console.log('Share cancelled'));
        } else {
            // Fallback to copying to clipboard
            navigator.clipboard.writeText(text).then(() => {
                alert('Analysis summary copied to clipboard!');
            });
        }
    }

    loadDemoContent() {
        // Load demo text
        document.getElementById('textInput').value = `The rapid advancement of artificial intelligence has fundamentally transformed numerous industries across the globe. From healthcare to finance, education to entertainment, AI technologies are reshaping how we work, learn, and interact with the world around us.

In the healthcare sector, AI-powered diagnostic tools are enhancing the accuracy of disease detection and treatment planning. Machine learning algorithms can analyze medical images with remarkable precision, often identifying patterns that might be overlooked by human practitioners. Furthermore, predictive analytics are enabling healthcare providers to anticipate patient needs and optimize resource allocation.

The financial industry has similarly embraced AI innovations. Automated trading systems leverage sophisticated algorithms to make split-second decisions based on market data analysis. Additionally, AI-driven fraud detection systems protect consumers by identifying suspicious transactions in real-time, significantly reducing financial crimes.

Education is another domain experiencing a profound AI-driven transformation. Personalized learning platforms adapt to individual student needs, providing customized content and pacing. Moreover, intelligent tutoring systems offer round-the-clock support, ensuring that learners receive assistance whenever they need it.

In conclusion, artificial intelligence continues to revolutionize various sectors, offering unprecedented opportunities for innovation and efficiency. As these technologies evolve, it is crucial to consider both their immense potential benefits and the ethical considerations they raise. The future undoubtedly holds even more transformative applications of AI across all aspects of human endeavor.`;
        
        // Update character count
        this.updateCharCount();
        
        // Auto-analyze after a short delay
        setTimeout(() => this.analyzeContent(), 500);
    }

    showPricing() {
        alert('Premium features coming soon! Get detailed AI model detection, section-by-section analysis, plagiarism source breakdown, and PDF reports.');
    }

    showError(message) {
        const errorEl = document.getElementById('errorMessage');
        errorEl.textContent = message;
        errorEl.style.display = 'block';
        errorEl.style.animation = 'shake 0.5s';
        
        setTimeout(() => {
            errorEl.style.animation = '';
        }, 500);
    }

    hideError() {
        document.getElementById('errorMessage').style.display = 'none';
    }
}

// Add additional styles for new elements
const additionalStyles = document.createElement('style');
additionalStyles.innerHTML = `
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .drag-over {
        border-color: var(--primary) !important;
        background: rgba(124, 58, 237, 0.1) !important;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        background: var(--light-gray);
        border-radius: 8px;
    }
    
    .metric-item label {
        display: block;
        font-size: 0.85rem;
        color: var(--gray);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        display: block;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .metric-item small {
        display: block;
        font-size: 0.75rem;
        color: var(--gray);
        margin-top: 0.25rem;
    }
    
    .stat-grid {
        margin-bottom: 1rem;
    }
    
    .stat-item {
        margin-bottom: 1rem;
    }
    
    .stat-item label {
        display: block;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .detected-model {
        text-align: center;
        padding: 2rem;
        background: var(--light-gray);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .detected-model i {
        font-size: 3rem;
        color: var(--primary);
        margin-bottom: 1rem;
    }
    
    .detected-model h4 {
        margin: 0;
        font-size: 1.25rem;
    }
    
    .signature-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: var(--light-gray);
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    
    .model-name {
        font-weight: 600;
        color: var(--primary);
    }
    
    .signature-type {
        font-size: 0.85rem;
        color: var(--gray);
    }
    
    .confidence {
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .section-item {
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--light-gray);
        border-radius: 8px;
    }
    
    .section-text {
        font-style: italic;
        color: var(--gray);
        margin: 0.5rem 0;
    }
    
    .section-score {
        font-weight: 600;
    }
`;
document.head.appendChild(additionalStyles);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.textAnalysisApp = new TextAnalysisApp();
});

// Add console branding
console.log('%cAI & Plagiarism Detection', 'font-size: 24px; font-weight: bold; color: #7c3aed;');
console.log('%cAdvanced Text Analysis Tool', 'font-size: 14px; color: #6b7280;');
console.log('%cPowered by Copyleaks & Custom Algorithms', 'font-size: 12px; color: #06b6d4;');
