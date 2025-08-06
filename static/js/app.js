// static/js/app.js - AI Detection Application Logic

class AIDetectionApp {
    constructor() {
        this.currentAnalysis = null;
        this.isPremium = false;
        this.currentTab = 'text';
        this.API_ENDPOINT = '/api/analyze';
        this.progressInterval = null;
        this.analysisStartTime = null;
        this.uploadedImage = null;
        
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
        // Tab switching
        window.switchTab = (tab) => {
            this.currentTab = tab;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tab === tab);
            });
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = content.id === `${tab}-tab` ? 'block' : 'none';
            });
        };

        // Image upload drag and drop
        const uploadArea = document.getElementById('imageUploadArea');
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
                    this.handleImageFile(files[0]);
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
        window.handleImageUpload = (event) => this.handleImageUpload(event);
        window.removeImage = () => this.removeImage();
    }

    showHowItWorks() {
        document.getElementById('howItWorksSection').style.display = 'flex';
    }

    hideHowItWorks() {
        document.getElementById('howItWorksSection').style.display = 'none';
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleImageFile(file);
        }
    }

    handleImageFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please upload a valid image file (JPG, PNG, WebP, or GIF)');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('Image size must be less than 16MB');
            return;
        }

        // Read and display image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.uploadedImage = {
                data: e.target.result.split(',')[1], // Remove data:image/jpeg;base64, prefix
                type: file.type,
                name: file.name
            };
            
            // Show preview
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('imagePreview').style.display = 'block';
            document.querySelector('.upload-label').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.uploadedImage = null;
        document.getElementById('imagePreview').style.display = 'none';
        document.querySelector('.upload-label').style.display = 'flex';
        document.getElementById('imageInput').value = '';
    }

    async analyzeContent() {
        this.analysisStartTime = Date.now();
        
        // Get input based on current tab
        let requestData = { is_pro: this.isPremium };
        
        if (this.currentTab === 'text') {
            const text = document.getElementById('textInput').value.trim();
            if (!text || text.length < 50) {
                this.showError('Please enter at least 50 characters of text');
                return;
            }
            requestData.text = text;
            
        } else if (this.currentTab === 'image') {
            if (!this.uploadedImage) {
                this.showError('Please upload an image to analyze');
                return;
            }
            requestData.image = this.uploadedImage.data;
            requestData.image_type = this.uploadedImage.type;
            
        } else if (this.currentTab === 'url') {
            const url = document.getElementById('urlInput').value.trim();
            if (!url || !this.isValidUrl(url)) {
                this.showError('Please enter a valid URL');
                return;
            }
            requestData.url = url;
        }

        // Reset UI
        this.hideError();
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('premiumAnalysis').style.display = 'none';
        document.getElementById('premiumCTA').style.display = 'block';
        
        // Show progress
        this.showProgress();
        
        // Disable analyze buttons
        const analyzeBtns = document.querySelectorAll('.analyze-btn');
        analyzeBtns.forEach(btn => {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        });

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
            console.log('AI Detection Analysis complete:', data);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'An error occurred during analysis');
            this.hideProgress();
        } finally {
            // Re-enable buttons
            analyzeBtns.forEach(btn => {
                btn.disabled = false;
                if (this.currentTab === 'text') {
                    btn.innerHTML = '<i class="fas fa-search"></i> <span>Analyze Text</span>';
                } else if (this.currentTab === 'image') {
                    btn.innerHTML = '<i class="fas fa-search"></i> <span>Analyze Image</span>';
                } else {
                    btn.innerHTML = '<i class="fas fa-search"></i> <span>Analyze</span>';
                }
            });
        }
    }

    showProgress() {
        const progressSection = document.getElementById('progressSection');
        progressSection.style.display = 'block';
        
        // Reset progress
        const stages = ['extract', 'patterns', 'perplexity', 'statistical', 'score'];
        let currentStage = 0;
        let methodCount = 0;
        
        // Animate stages
        this.progressInterval = setInterval(() => {
            if (currentStage < stages.length) {
                // Mark current stage as active
                document.querySelectorAll('.stage').forEach((stage, index) => {
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
                
                // Update method count
                if (currentStage === 2) {
                    methodCount = Math.min(methodCount + 1, 5);
                    document.getElementById('methodCount').textContent = methodCount;
                }
                
                currentStage++;
            }
        }, 800);
    }

    hideProgress() {
        const progressSection = document.getElementById('progressSection');
        progressSection.style.display = 'none';
        
        // Clear interval
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Reset stages
        document.querySelectorAll('.stage').forEach(stage => {
            stage.classList.remove('active', 'complete');
        });
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

        // Display AI probability with animation
        const aiProbability = data.ai_probability || 50;
        this.animateAIProbability(aiProbability);
        
        // Display content info
        document.getElementById('contentType').textContent = data.content_type === 'text' ? 'Text Content' : 'Image Content';
        document.getElementById('contentLength').textContent = data.content_type === 'text' ? 
            `${data.word_count || 0} words` : 'Image analyzed';
        document.getElementById('analysisTime').textContent = `${analysisTime}s`;
        document.getElementById('confidenceLevel').textContent = `${data.confidence_level || 'Unknown'} confidence`;
        
        // Display summary
        document.getElementById('analysisSummary').textContent = data.summary || 'Analysis complete';
        
        // Update quick stats
        const patternCount = data.pattern_analysis?.detected_patterns?.length || 0;
        document.getElementById('patternCount').textContent = patternCount;
        document.getElementById('methodsUsed').textContent = '5+';
        document.getElementById('accuracyRate').textContent = Math.round(this.calculateConfidence(data));
        
        // Display detection breakdown
        document.getElementById('detectionBreakdown').innerHTML = 
            this.createDetectionBreakdown(data);
        
        // Create indicators summary
        this.createIndicatorsSummary(data);
        
        // If premium, show all analysis
        if (this.isPremium && data.is_pro) {
            this.displayPremiumAnalysis(data);
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

    createDetectionBreakdown(data) {
        let html = '';
        
        // Pattern Analysis
        if (data.pattern_analysis) {
            const score = data.pattern_analysis.ai_patterns_score || 0;
            html += this.createDetectionFactor(
                'Pattern Analysis',
                score,
                'Detected AI writing patterns and phrases',
                'brain'
            );
        }
        
        // Perplexity Analysis
        if (data.perplexity_analysis) {
            const score = data.perplexity_analysis.ai_probability || 0;
            html += this.createDetectionFactor(
                'Perplexity Analysis',
                score,
                'Text predictability and complexity metrics',
                'chart-area'
            );
        }
        
        // Statistical Analysis
        if (data.statistical_analysis) {
            const score = data.statistical_analysis.ai_probability || 0;
            html += this.createDetectionFactor(
                'Statistical Analysis',
                score,
                'Word distribution and sentence patterns',
                'calculator'
            );
        }
        
        // Copyleaks (if available)
        if (data.copyleaks_analysis) {
            const score = data.copyleaks_analysis.ai_probability || 0;
            html += this.createDetectionFactor(
                'Copyleaks AI Detection',
                score,
                'Enterprise-grade AI detection algorithm',
                'shield-alt'
            );
        }
        
        return html;
    }

    createDetectionFactor(name, score, description, icon) {
        const fillColor = this.getScoreColor(score);
        
        return `
            <div class="trust-factor">
                <div class="factor-header">
                    <div class="factor-info">
                        <i class="fas fa-${icon}"></i>
                        <span>${name}</span>
                    </div>
                    <span class="factor-score" style="color: ${fillColor}">${score}%</span>
                </div>
                <div class="factor-bar">
                    <div class="factor-fill" style="width: ${score}%; background: ${fillColor}"></div>
                </div>
                <p class="factor-description">${description}</p>
            </div>
        `;
    }

    getScoreColor(score) {
        if (score >= 70) return '#ef4444';
        if (score >= 50) return '#f59e0b';
        if (score >= 30) return '#3b82f6';
        return '#10b981';
    }

    createIndicatorsSummary(data) {
        const summaryEl = document.getElementById('indicatorsSummary');
        let html = '<ul>';
        
        // AI probability summary
        const aiProb = data.ai_probability || 50;
        if (aiProb >= 70) {
            html += '<li><i class="fas fa-exclamation-circle text-danger"></i> <strong>High AI probability:</strong> Multiple strong indicators of AI generation detected.</li>';
        } else if (aiProb >= 50) {
            html += '<li><i class="fas fa-exclamation-triangle text-warning"></i> <strong>Moderate AI probability:</strong> Several AI patterns found, suggesting AI involvement.</li>';
        } else {
            html += '<li><i class="fas fa-check-circle text-success"></i> <strong>Low AI probability:</strong> Content appears predominantly human-created.</li>';
        }
        
        // Pattern findings
        if (data.pattern_analysis?.detected_patterns?.length > 0) {
            const patternCount = data.pattern_analysis.detected_patterns.length;
            html += `<li><i class="fas fa-brain text-info"></i> Detected <strong>${patternCount} AI writing patterns</strong> including formal transitions and common AI phrases.</li>`;
        }
        
        // Perplexity findings
        if (data.perplexity_analysis) {
            const perplexity = data.perplexity_analysis.perplexity || 0;
            if (perplexity < 50) {
                html += '<li><i class="fas fa-chart-line text-warning"></i> <strong>Low text perplexity:</strong> Unusually predictable text structure typical of AI.</li>';
            }
        }
        
        // Statistical findings
        if (data.statistical_analysis?.vocabulary_diversity) {
            const diversity = data.statistical_analysis.vocabulary_diversity;
            if (diversity < 0.3) {
                html += '<li><i class="fas fa-calculator text-warning"></i> <strong>Limited vocabulary diversity:</strong> Repetitive word usage patterns detected.</li>';
            }
        }
        
        html += '</ul>';
        summaryEl.innerHTML = html;
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
        
        // Show premium analysis
        this.displayPremiumAnalysis(this.currentAnalysis);
    }

    displayPremiumAnalysis(data) {
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
        
        // Forensics Card (for images)
        if (data.forensics_analysis) {
            cards.push(this.createForensicsCard(data));
        }
        
        // Detection Breakdown Card
        if (data.detection_breakdown) {
            cards.push(this.createBreakdownCard(data));
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

    createForensicsCard(data) {
        const forensics = data.forensics_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-microscope"></i> Image Forensics</h3>
                    <span class="card-score ${this.getScoreClass(forensics.ai_probability || 0)}">${forensics.ai_probability || 0}%</span>
                </div>
                <div class="card-content">
                    <div class="forensics-results">
                        ${forensics.artifacts_detected ? `
                            <div class="artifacts-found">
                                <h5>AI Generation Artifacts:</h5>
                                <ul>
                                    ${forensics.artifacts_detected.map(artifact => 
                                        `<li><i class="fas fa-search"></i> ${artifact}</li>`
                                    ).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        ${forensics.metadata_analysis ? `
                            <div class="metadata-info">
                                <h5>Metadata Analysis:</h5>
                                <p>${forensics.metadata_analysis}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    createBreakdownCard(data) {
        const breakdown = data.detection_breakdown || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-chart-pie"></i> Detection Breakdown</h3>
                </div>
                <div class="card-content">
                    ${breakdown.detection_methods ? `
                        <div class="method-weights">
                            <h5>Method Contributions:</h5>
                            ${breakdown.detection_methods.map(method => `
                                <div class="method-item">
                                    <span class="method-name">${method.method}</span>
                                    <span class="method-score">${method.score}%</span>
                                    <span class="method-weight">(${method.weight} weight)</span>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                    ${breakdown.confidence_factors ? `
                        <div class="confidence-factors">
                            <h5>Confidence Factors:</h5>
                            <ul>
                                ${breakdown.confidence_factors.map(factor => 
                                    `<li>${factor}</li>`
                                ).join('')}
                            </ul>
                        </div>
                    ` : ''}
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
                body: JSON.stringify(this.currentAnalysis)
            });
            
            if (!response.ok) throw new Error('PDF generation failed');
            
            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ai-detection-report-${Date.now()}.pdf`;
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
        
        const aiProb = this.currentAnalysis.ai_probability || 50;
        const text = `AI Detection Result: ${aiProb}% probability of AI generation. Analyzed with 5+ detection methods.`;
        
        if (navigator.share) {
            navigator.share({
                title: 'AI Detection Analysis',
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
        
        // Switch to text tab
        this.currentTab = 'text';
        window.switchTab('text');
        
        // Auto-analyze
        setTimeout(() => this.analyzeContent(), 500);
    }

    showPricing() {
        alert('Premium features coming soon! Get advanced AI model detection, Copyleaks integration, and detailed PDF reports.');
    }

    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
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

// Add shake animation
const shakeStyle = document.createElement('style');
shakeStyle.innerHTML = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
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
    
    .pattern-description {
        font-size: 0.85rem;
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
    
    .method-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--light-gray);
    }
    
    .method-name {
        font-weight: 600;
    }
    
    .method-score {
        color: var(--primary);
        font-weight: 700;
    }
    
    .method-weight {
        font-size: 0.85rem;
        color: var(--gray);
    }
`;
document.head.appendChild(shakeStyle);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.aiDetectionApp = new AIDetectionApp();
});

// Add console branding
console.log('%cAI Detection Analyzer', 'font-size: 24px; font-weight: bold; color: #7c3aed;');
console.log('%cAdvanced AI Content Detection', 'font-size: 14px; color: #6b7280;');
console.log('%cPowered by 5+ Detection Methods', 'font-size: 12px; color: #06b6d4;');
