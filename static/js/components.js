// static/js/components.js - Reusable UI Components for AI Detection

const analysisComponents = {
    /**
     * Create trust score gauge chart
     */
    createTrustScoreGauge: function(canvasId, score) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
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
        
        // Create gradient based on score
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        if (score >= 80) {
            gradient.addColorStop(0, '#10b981');
            gradient.addColorStop(1, '#059669');
        } else if (score >= 60) {
            gradient.addColorStop(0, '#3b82f6');
            gradient.addColorStop(1, '#2563eb');
        } else if (score >= 40) {
            gradient.addColorStop(0, '#f59e0b');
            gradient.addColorStop(1, '#d97706');
        } else {
            gradient.addColorStop(0, '#ef4444');
            gradient.addColorStop(1, '#dc2626');
        }
        
        ctx.strokeStyle = gradient;
        ctx.stroke();
        
        // Draw score text
        ctx.font = 'bold 36px Arial';
        ctx.fillStyle = '#1f2937';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(score), centerX, centerY - 10);
        
        // Draw label
        ctx.font = '14px Arial';
        ctx.fillStyle = '#6b7280';
        ctx.fillText('Trust Score', centerX, centerY + 15);
    },

    /**
     * Create trust breakdown factors
     */
    createTrustBreakdown: function(data) {
        const factors = [
            {
                name: 'Author Credibility',
                score: data.author_analysis?.credibility_score || 50,
                icon: 'user-check',
                description: 'Verification of author credentials and history'
            },
            {
                name: 'Source Reliability',
                score: this._convertRatingToScore(data.source_credibility?.rating),
                icon: 'building',
                description: 'Domain reputation and fact-checking standards'
            },
            {
                name: 'Bias Detection',
                score: 100 - (data.bias_analysis?.overall_bias || 50),
                icon: 'balance-scale',
                description: 'Political and ideological neutrality assessment'
            },
            {
                name: 'Fact Accuracy',
                score: this._calculateFactScore(data.fact_checks),
                icon: 'check-circle',
                description: 'Verification of claims against trusted sources'
            },
            {
                name: 'Transparency',
                score: data.transparency_analysis?.transparency_score || 50,
                icon: 'eye',
                description: 'Clear sourcing and disclosure practices'
            },
            {
                name: 'Manipulation Check',
                score: 100 - (data.persuasion_analysis?.manipulation_score || 50),
                icon: 'masks-theater',
                description: 'Detection of emotional manipulation tactics'
            }
        ];

        let html = '';
        factors.forEach((factor, index) => {
            const fillColor = this._getScoreColor(factor.score);
            html += `
                <div class="trust-factor" style="animation-delay: ${index * 0.1}s">
                    <div class="factor-header">
                        <div class="factor-info">
                            <i class="fas fa-${factor.icon}"></i>
                            <span>${factor.name}</span>
                        </div>
                        <span class="factor-score" style="color: ${fillColor}">${factor.score}</span>
                    </div>
                    <div class="factor-bar">
                        <div class="factor-fill" style="width: ${factor.score}%; background: ${fillColor}; animation: fillBar 1s ease-out ${index * 0.1}s both"></div>
                    </div>
                    <p class="factor-description">${factor.description}</p>
                </div>
            `;
        });

        return html;
    },

    /**
     * Create author analysis card
     */
    createAuthorCard: function(data) {
        const author = data.author_analysis || {};
        const article = data.article || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-user-check"></i> Author Analysis</h3>
                    <span class="card-score ${this._getScoreClass(author.credibility_score)}">${author.credibility_score || 0}/100</span>
                </div>
                <div class="card-content">
                    <div class="author-profile">
                        <div class="author-avatar">
                            <i class="fas fa-user-circle"></i>
                        </div>
                        <div class="author-info">
                            <h4>${article.author || 'Unknown Author'}</h4>
                            <p class="author-bio">${author.bio || 'No biographical information available'}</p>
                            ${author.verified ? `
                                <div class="verification-badges">
                                    ${author.journalist ? '<span class="badge journalist">Professional Journalist</span>' : ''}
                                    ${author.verified ? '<span class="badge verified">Verified</span>' : ''}
                                    ${author.staff_writer ? '<span class="badge staff">Staff Writer</span>' : ''}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    
                    ${author.professional_info ? `
                        <div class="professional-info">
                            <h5>Professional Background</h5>
                            <p><strong>Experience:</strong> ${author.professional_info.years_experience || 'Unknown'} years</p>
                            <p><strong>Specialties:</strong> ${author.professional_info.specialties?.join(', ') || 'Not specified'}</p>
                            ${author.professional_info.awards ? `<p><strong>Awards:</strong> ${author.professional_info.awards.join(', ')}</p>` : ''}
                        </div>
                    ` : ''}
                    
                    ${author.verification_status ? `
                        <div class="verification-status">
                            <h5>Verification Details</h5>
                            ${Object.entries(author.verification_status).map(([source, verified]) => `
                                <p><i class="fas ${verified ? 'fa-check-circle text-success' : 'fa-times-circle text-danger'}"></i> ${source}</p>
                            `).join('')}
                        </div>
                    ` : ''}
                    
                    ${author.social_media ? `
                        <div class="online-presence">
                            <h5>Online Presence</h5>
                            <div class="social-links">
                                ${author.social_media.twitter ? `<a href="${author.social_media.twitter}" class="social-link" target="_blank"><i class="fab fa-twitter"></i></a>` : ''}
                                ${author.social_media.linkedin ? `<a href="${author.social_media.linkedin}" class="social-link" target="_blank"><i class="fab fa-linkedin"></i></a>` : ''}
                                ${author.social_media.website ? `<a href="${author.social_media.website}" class="social-link" target="_blank"><i class="fas fa-globe"></i></a>` : ''}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="credibility-explanation">
                        <h5>Credibility Assessment</h5>
                        <p>${this._getCredibilityExplanation(author.credibility_score)}</p>
                        ${author.red_flags && author.red_flags.length > 0 ? `
                            <p class="text-danger"><strong>Red Flags:</strong> ${author.red_flags.join(', ')}</p>
                        ` : ''}
                        <p class="advice"><i class="fas fa-info-circle"></i> ${this._getAuthorAdvice(author.credibility_score)}</p>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Create bias analysis card
     */
    createBiasCard: function(data) {
        const bias = data.bias_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-balance-scale"></i> Bias Analysis</h3>
                    <span class="card-score ${this._getScoreClass(100 - (bias.overall_bias || 50))}">${bias.overall_bias || 50}% Biased</span>
                </div>
                <div class="card-content">
                    <div class="bias-summary">
                        <h5>Overall Political Lean</h5>
                        <p class="bias-label">${this._getPoliticalLeanLabel(bias.political_lean)}</p>
                    </div>
                    
                    <div class="bias-dimensions">
                        <h5>Bias Dimensions</h5>
                        ${this._createBiasDimension('Political Bias', bias.political_bias, -100, 100, 'Left', 'Right')}
                        ${this._createBiasDimension('Corporate Bias', bias.corporate_bias, 0, 100, 'None', 'High')}
                        ${this._createBiasDimension('Sensationalism', bias.sensationalism, 0, 100, 'Factual', 'Sensational')}
                        ${this._createBiasDimension('Establishment Bias', bias.establishment_bias, -100, 100, 'Anti', 'Pro')}
                        ${this._createBiasDimension('Nationalistic Bias', bias.nationalistic_bias, 0, 100, 'Global', 'Nationalistic')}
                    </div>
                    
                    <div class="bias-visualization">
                        <canvas id="biasRadarChart" width="300" height="200"></canvas>
                    </div>
                    
                    ${bias.manipulation_tactics ? `
                        <div class="manipulation-section">
                            <h5>Detected Manipulation Tactics</h5>
                            <div class="tactics-grid">
                                ${bias.manipulation_tactics.map(tactic => `
                                    <div class="tactic-chip">
                                        <span class="tactic-name">${tactic}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${bias.loaded_phrases ? `
                        <div class="loaded-phrases">
                            <h5>Emotionally Loaded Language</h5>
                            ${bias.loaded_phrases.slice(0, 3).map(phrase => `
                                <div class="phrase-item">
                                    <span class="phrase-text">"${phrase.text}"</span>
                                    <span class="phrase-type">${phrase.type}</span>
                                </div>
                            `).join('')}
                            ${bias.loaded_phrases.length > 3 ? `<p class="text-muted">+${bias.loaded_phrases.length - 3} more instances</p>` : ''}
                        </div>
                    ` : ''}
                    
                    <div class="ai-insight">
                        <h5>AI Insight</h5>
                        <p>${this._getBiasInsight(bias)}</p>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Create fact checking card
     */
    createFactCheckCard: function(data) {
        const factChecks = data.fact_checks || [];
        const verifiedCount = factChecks.filter(f => f.verdict === 'TRUE').length;
        const falseCount = factChecks.filter(f => f.verdict === 'FALSE').length;
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-fact-check"></i> Fact Verification</h3>
                    <span class="card-score">${verifiedCount}/${factChecks.length} Verified</span>
                </div>
                <div class="card-content">
                    <div class="fact-summary">
                        <strong>${verifiedCount}</strong> claims verified as true, 
                        <strong>${falseCount}</strong> found false, 
                        <strong>${factChecks.length - verifiedCount - falseCount}</strong> unverifiable
                    </div>
                    
                    <div class="fact-checks">
                        ${factChecks.slice(0, 5).map(fact => `
                            <div class="fact-check-item ${fact.verdict.toLowerCase()}">
                                <p class="fact-claim">"${fact.claim}"</p>
                                <div class="fact-verdict">
                                    <span class="verdict-label ${fact.verdict.toLowerCase()}">${fact.verdict}</span>
                                    <span class="confidence">${fact.confidence}% confidence</span>
                                </div>
                                ${fact.explanation ? `<p class="fact-explanation">${fact.explanation}</p>` : ''}
                                ${fact.sources_checked ? `
                                    <div class="fact-sources-checked">
                                        <small>Checked: ${fact.sources_checked.join(', ')}</small>
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                        ${factChecks.length > 5 ? `<p class="more-facts">+${factChecks.length - 5} more claims analyzed</p>` : ''}
                    </div>
                    
                    ${data.fact_check_sources ? `
                        <div class="fact-sources">
                            <h5>Verification Sources Used</h5>
                            <div class="source-badges">
                                ${data.fact_check_sources.map(source => `
                                    <span class="source-badge ${source.verified ? 'verified' : ''}">${source.name}</span>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Create source credibility card
     */
    createSourceCard: function(data) {
        const source = data.source_credibility || {};
        const article = data.article || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-building"></i> Source Credibility</h3>
                    <span class="card-score ${this._getScoreClass(this._convertRatingToScore(source.rating))}">${source.rating || 'Unknown'}</span>
                </div>
                <div class="card-content">
                    <div class="source-info">
                        <h4>${article.domain || 'Unknown Source'}</h4>
                        <p class="source-type">${source.type || 'Website'}</p>
                    </div>
                    
                    <div class="credibility-details">
                        <div class="detail-item">
                            <label>Fact-Checking Record</label>
                            <span>${source.fact_check_rating || 'Not rated'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Media Bias Rating</label>
                            <span>${source.bias_rating || 'Not rated'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Ownership</label>
                            <p>${source.ownership || 'Unknown'}</p>
                        </div>
                        <div class="detail-item">
                            <label>Founded</label>
                            <span>${source.founded || 'Unknown'}</span>
                        </div>
                        ${source.press_freedom_score ? `
                            <div class="detail-item">
                                <label>Press Freedom Score</label>
                                <span>${source.press_freedom_score}/100</span>
                            </div>
                        ` : ''}
                    </div>
                    
                    ${source.credibility_notes ? `
                        <div class="credibility-explanation">
                            <h5>Credibility Notes</h5>
                            <p>${source.credibility_notes}</p>
                        </div>
                    ` : ''}
                    
                    ${source.awards ? `
                        <div class="source-awards">
                            <h5>Awards & Recognition</h5>
                            <ul>
                                ${source.awards.map(award => `<li>${award}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Create transparency analysis card
     */
    createTransparencyCard: function(data) {
        const transparency = data.transparency_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-eye"></i> Transparency Analysis</h3>
                    <span class="card-score ${this._getScoreClass(transparency.transparency_score)}">${transparency.transparency_score || 0}/100</span>
                </div>
                <div class="card-content">
                    <div class="transparency-indicators">
                        ${this._createTransparencyIndicator('Author Attribution', transparency.has_author)}
                        ${this._createTransparencyIndicator('Publication Date', transparency.has_date)}
                        ${this._createTransparencyIndicator('Source Citations', transparency.has_sources)}
                        ${this._createTransparencyIndicator('Correction Policy', transparency.has_corrections)}
                        ${this._createTransparencyIndicator('Funding Disclosure', transparency.has_funding)}
                        ${this._createTransparencyIndicator('Conflict Disclosure', transparency.has_conflicts)}
                    </div>
                    
                    ${transparency.missing_elements && transparency.missing_elements.length > 0 ? `
                        <div class="missing-elements">
                            <h5>Missing Transparency Elements</h5>
                            <ul>
                                ${transparency.missing_elements.map(element => `
                                    <li><i class="fas fa-exclamation-triangle text-warning"></i> ${element}</li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${transparency.found_disclosures ? `
                        <div class="found-indicators">
                            <h5>Transparency Strengths</h5>
                            <ul>
                                ${transparency.found_disclosures.map(disclosure => `
                                    <li><i class="fas fa-check-circle text-success"></i> ${disclosure}</li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Create manipulation detection card
     */
    createManipulationCard: function(data) {
        const manipulation = data.persuasion_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-masks-theater"></i> Manipulation Detection</h3>
                    <span class="card-score ${this._getScoreClass(100 - (manipulation.manipulation_score || 50))}">${manipulation.manipulation_score || 0}% Manipulative</span>
                </div>
                <div class="card-content">
                    <div class="manipulation-level">
                        <h5>Manipulation Level</h5>
                        <p class="level-${(manipulation.manipulation_level || 'unknown').toLowerCase()}">${manipulation.manipulation_level || 'Unknown'}</p>
                    </div>
                    
                    ${manipulation.detected_tactics && manipulation.detected_tactics.length > 0 ? `
                        <div class="tactics-list">
                            <h5>Detected Tactics</h5>
                            ${manipulation.detected_tactics.map(tactic => `
                                <div class="tactic-item severity-${tactic.severity}">
                                    <strong>${tactic.name}</strong>
                                    <p>${tactic.description}</p>
                                    ${tactic.examples ? `
                                        <div class="examples">
                                            ${tactic.examples.slice(0, 2).map(ex => `<small>"${ex}"</small>`).join('')}
                                        </div>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                    
                    ${manipulation.emotional_triggers ? `
                        <div class="emotional-triggers">
                            <h5>Emotional Triggers Used</h5>
                            <div class="keywords">
                                ${manipulation.emotional_triggers.map(trigger => `
                                    <span class="keyword">${trigger}</span>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Create content analysis card
     */
    createContentCard: function(data) {
        const content = data.content_analysis || {};
        
        return `
            <div class="analysis-card">
                <div class="card-header">
                    <h3><i class="fas fa-file-alt"></i> Content Analysis</h3>
                </div>
                <div class="card-content">
                    <div class="content-metrics">
                        <div class="metric">
                            <span class="metric-value">${content.readability_score || 0}</span>
                            <span class="metric-label">Readability</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">${content.complexity || 'Medium'}</span>
                            <span class="metric-label">Complexity</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">${content.sentiment || 'Neutral'}</span>
                            <span class="metric-label">Sentiment</span>
                        </div>
                    </div>
                    
                    ${content.readability_analysis ? `
                        <div class="readability-section">
                            <h5>Readability Analysis</h5>
                            <div class="readability-score">
                                <span class="score-value">${content.readability_analysis.flesch_score || 0}</span>
                                <span class="score-label">Flesch Reading Ease</span>
                            </div>
                            <p class="target-audience">Target audience: ${content.readability_analysis.target_audience || 'General public'}</p>
                        </div>
                    ` : ''}
                    
                    ${content.quality_indicators ? `
                        <div class="quality-indicators">
                            <h5>Content Quality Indicators</h5>
                            <div class="indicators-grid">
                                <div class="indicator-item">
                                    <span class="indicator-label">Grammar Score</span>
                                    <span class="indicator-value">${content.quality_indicators.grammar_score || 0}%</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Originality</span>
                                    <span class="indicator-value">${content.quality_indicators.originality || 0}%</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Source Diversity</span>
                                    <span class="indicator-value">${content.quality_indicators.source_diversity || 'Low'}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Evidence Quality</span>
                                    <span class="indicator-value">${content.quality_indicators.evidence_quality || 'Fair'}</span>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Create bias visualization radar chart
     */
    createBiasVisualization: function(biasData) {
        const canvas = document.getElementById('biasRadarChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Data for radar chart
        const data = {
            labels: ['Political', 'Corporate', 'Sensational', 'Establishment', 'Nationalistic'],
            datasets: [{
                label: 'Bias Levels',
                data: [
                    Math.abs(biasData.political_bias || 0),
                    biasData.corporate_bias || 0,
                    biasData.sensationalism || 0,
                    Math.abs(biasData.establishment_bias || 0),
                    biasData.nationalistic_bias || 0
                ],
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(99, 102, 241, 1)'
            }]
        };
        
        // Create radar chart
        new Chart(ctx, {
            type: 'radar',
            data: data,
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    },

    // Helper methods
    _convertRatingToScore: function(rating) {
        const ratingMap = {
            'Very High': 95,
            'High': 80,
            'Medium': 60,
            'Low': 40,
            'Very Low': 20,
            'Unknown': 50
        };
        return ratingMap[rating] || 50;
    },

    _calculateFactScore: function(factChecks) {
        if (!factChecks || factChecks.length === 0) return 50;
        
        const verified = factChecks.filter(f => f.verdict === 'TRUE').length;
        return Math.round((verified / factChecks.length) * 100);
    },

    _getScoreColor: function(score) {
        if (score >= 80) return '#10b981';
        if (score >= 60) return '#3b82f6';
        if (score >= 40) return '#f59e0b';
        return '#ef4444';
    },

    _getScoreClass: function(score) {
        if (score >= 80) return
