"""
Image AI Analyzer Service
Detects AI-generated images through various analysis methods
"""
import os
import logging
import base64
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from io import BytesIO
import json
import math

# Try to import image processing libraries
try:
    from PIL import Image, ImageStat, ImageFilter, ImageChops
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    np = None

# Use TYPE_CHECKING for type hints that might not be available at runtime
if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

if not PIL_AVAILABLE:
    logger.warning("PIL/Pillow not available. Image analysis will be limited.")

class ImageAIAnalyzer:
    """Analyzes images for AI generation artifacts"""
    
    def __init__(self):
        self.pil_available = PIL_AVAILABLE
        
        # Known AI model signatures and patterns
        self.ai_signatures = {
            'midjourney': {
                'patterns': ['perfect symmetry', 'unrealistic lighting', 'texture artifacts'],
                'metadata_keys': ['midjourney', 'mj', 'discord']
            },
            'dalle': {
                'patterns': ['grid artifacts', 'unnatural edges', 'color banding'],
                'metadata_keys': ['openai', 'dalle', 'dall-e', 'bing']
            },
            'stable_diffusion': {
                'patterns': ['noise patterns', 'latent artifacts', 'repetitive textures'],
                'metadata_keys': ['stable-diffusion', 'sd', 'stability', 'automatic1111', 'comfyui']
            },
            'leonardo': {
                'patterns': ['smooth gradients', 'perfect lighting'],
                'metadata_keys': ['leonardo', 'leonardo.ai']
            },
            'adobe_firefly': {
                'patterns': ['clean edges', 'professional finish'],
                'metadata_keys': ['adobe', 'firefly', 'creative cloud']
            }
        }
    
    def analyze(self, image_data: str, image_type: str = 'image/jpeg') -> Dict[str, Any]:
        """Analyze image for AI generation"""
        try:
            if not self.pil_available:
                return self._basic_analysis(image_data)
            
            # Decode base64 image
            image = self._decode_image(image_data)
            if not image:
                return {
                    'ai_probability': 50,
                    'error': 'Failed to decode image'
                }
            
            # Log image info
            logger.info(f"Analyzing image: {image.size}, mode: {image.mode}")
            
            # Perform various analyses
            metadata_analysis = self._analyze_metadata(image)
            pixel_analysis = self._analyze_pixels(image)
            frequency_analysis = self._analyze_frequency_domain(image)
            artifact_analysis = self._detect_ai_artifacts(image)
            
            # Calculate overall AI probability
            ai_probability = self._calculate_image_ai_probability(
                metadata_analysis,
                pixel_analysis,
                frequency_analysis,
                artifact_analysis
            )
            
            return {
                'ai_probability': round(ai_probability, 1),
                'metadata_analysis': metadata_analysis,
                'pixel_analysis': pixel_analysis,
                'frequency_analysis': frequency_analysis,
                'artifact_analysis': artifact_analysis,
                'detected_model': self._detect_model(metadata_analysis, artifact_analysis, pixel_analysis),
                'summary': self._create_image_summary(ai_probability, artifact_analysis)
            }
            
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}", exc_info=True)
            return {
                'ai_probability': 50,
                'error': str(e)
            }
    
    def forensic_analysis(self, image_data: str) -> Dict[str, Any]:
        """Perform deep forensic analysis (Pro feature)"""
        try:
            if not self.pil_available:
                return {'error': 'Forensic analysis requires PIL'}
            
            image = self._decode_image(image_data)
            if not image:
                return {'error': 'Failed to decode image'}
            
            # Error level analysis
            ela_results = self._error_level_analysis(image)
            
            # Compression artifact analysis
            compression_analysis = self._analyze_compression_artifacts(image)
            
            # Color distribution analysis
            color_analysis = self._analyze_color_distribution(image)
            
            # Edge coherence analysis
            edge_analysis = self._analyze_edge_coherence(image)
            
            # Calculate forensic AI probability
            ai_probability = self._calculate_forensic_probability(
                ela_results, compression_analysis, color_analysis, edge_analysis
            )
            
            return {
                'ai_probability': round(ai_probability, 1),
                'ela_anomalies': ela_results.get('anomalies', []),
                'compression_artifacts': compression_analysis.get('ai_artifacts', []),
                'color_anomalies': color_analysis.get('anomalies', []),
                'edge_inconsistencies': edge_analysis.get('inconsistencies', []),
                'artifacts_detected': self._summarize_artifacts(
                    ela_results, compression_analysis, color_analysis, edge_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Forensic analysis error: {str(e)}", exc_info=True)
            return {
                'ai_probability': 50,
                'error': str(e)
            }
    
    def detect_generation_model(self, image_data: str) -> Dict[str, Any]:
        """Attempt to detect which AI model generated the image"""
        try:
            if not self.pil_available:
                return {
                    'detected_model': 'Unknown',
                    'confidence': 0
                }
            
            image = self._decode_image(image_data)
            if not image:
                return {
                    'detected_model': 'Unknown',
                    'confidence': 0
                }
            
            # Check for model-specific signatures
            signatures_found = []
            
            # Analyze patterns
            for model, info in self.ai_signatures.items():
                score = 0
                
                # Check metadata
                if hasattr(image, 'info'):
                    metadata_str = str(image.info).lower()
                    for key in info['metadata_keys']:
                        if key in metadata_str:
                            score += 40
                            break
                
                # Check visual patterns
                pattern_score = self._check_model_patterns(image, model, info['patterns'])
                score += pattern_score
                
                if score > 30:
                    signatures_found.append({
                        'model': model,
                        'confidence': min(score, 90)
                    })
            
            # Sort by confidence
            signatures_found.sort(key=lambda x: x['confidence'], reverse=True)
            
            if signatures_found:
                return {
                    'detected_model': signatures_found[0]['model'].replace('_', ' ').title(),
                    'confidence': signatures_found[0]['confidence'],
                    'all_signatures': signatures_found
                }
            else:
                return {
                    'detected_model': 'Unknown AI Model',
                    'confidence': 50
                }
                
        except Exception as e:
            logger.error(f"Model detection error: {str(e)}", exc_info=True)
            return {
                'detected_model': 'Unknown',
                'confidence': 0,
                'error': str(e)
            }
    
    def detect_ai_artifacts(self, image_data: str) -> Dict[str, Any]:
        """Detect specific AI generation artifacts"""
        try:
            if not self.pil_available:
                return {'ai_score': 50}
            
            image = self._decode_image(image_data)
            if not image:
                return {'ai_score': 50}
            
            artifacts = []
            ai_score = 0
            
            # Check for perfect symmetry (common in AI)
            symmetry_score = self._check_symmetry(image)
            if symmetry_score > 0.9:
                artifacts.append("Perfect symmetry detected")
                ai_score += 20
            elif symmetry_score > 0.8:
                artifacts.append("High symmetry detected")
                ai_score += 10
            
            # Check for unrealistic smoothness
            smoothness = self._check_smoothness(image)
            if smoothness > 0.85:
                artifacts.append("Unrealistic surface smoothness")
                ai_score += 15
            elif smoothness > 0.75:
                artifacts.append("Unusual smoothness detected")
                ai_score += 8
            
            # Check for repetitive patterns
            repetition = self._check_repetitive_patterns(image)
            if repetition > 0.7:
                artifacts.append("Repetitive pattern artifacts")
                ai_score += 15
            elif repetition > 0.5:
                artifacts.append("Some pattern repetition detected")
                ai_score += 8
            
            # Check for edge artifacts
            edge_artifacts, edge_score = self._check_edge_artifacts(image)
            if edge_artifacts:
                artifacts.extend(edge_artifacts)
                ai_score += edge_score
            
            # Check for color banding
            banding_score = self._check_color_banding(image)
            if banding_score > 0.6:
                artifacts.append("Color banding detected")
                ai_score += 10
            
            # Check for noise patterns
            noise_score = self._check_noise_patterns(image)
            if noise_score > 0.7:
                artifacts.append("Artificial noise patterns")
                ai_score += 10
            
            return {
                'ai_score': min(ai_score, 90),
                'artifacts': artifacts,
                'artifact_count': len(artifacts)
            }
            
        except Exception as e:
            logger.error(f"Artifact detection error: {str(e)}", exc_info=True)
            return {'ai_score': 50, 'error': str(e)}
    
    def _decode_image(self, image_data: str) -> Optional[Any]:
        """Decode base64 image data"""
        if not self.pil_available or not Image:
            return None
            
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            return image
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return None
    
    def _analyze_metadata(self, image: Any) -> Dict[str, Any]:
        """Analyze image metadata for AI signatures"""
        metadata = {}
        ai_indicators = 0
        detected_software = []
        
        if hasattr(image, 'info'):
            metadata = image.info
            
            # Check for AI-related metadata
            metadata_str = str(metadata).lower()
            ai_keywords = [
                'ai', 'generated', 'midjourney', 'dalle', 'dall-e', 'stable diffusion',
                'artificial', 'synthesis', 'gan', 'neural', 'automatic1111',
                'comfyui', 'leonardo', 'firefly', 'runway', 'wombo', 'nightcafe'
            ]
            
            for keyword in ai_keywords:
                if keyword in metadata_str:
                    ai_indicators += 1
                    detected_software.append(keyword)
            
            # Check EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                # Check for AI software in EXIF
                software_tag = 0x0131  # Software EXIF tag
                if software_tag in exif:
                    software = str(exif[software_tag]).lower()
                    for keyword in ai_keywords:
                        if keyword in software:
                            ai_indicators += 2
                            detected_software.append(f"EXIF: {keyword}")
        
        return {
            'has_metadata': bool(metadata),
            'ai_indicators': ai_indicators,
            'suspicious_metadata': ai_indicators > 0,
            'detected_software': list(set(detected_software))
        }
    
    def _analyze_pixels(self, image: Any) -> Dict[str, Any]:
        """Analyze pixel-level characteristics"""
        if not self.pil_available or not ImageStat:
            return {}
            
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image statistics
            stat = ImageStat.Stat(image)
            
            # Analyze color distribution
            mean = stat.mean
            stddev = stat.stddev
            
            # Check for unusual uniformity (common in AI)
            uniformity_score = 0
            for std in stddev:
                if std < 20:  # Very low standard deviation
                    uniformity_score += 0.33
                elif std < 30:
                    uniformity_score += 0.16
            
            # Check for perfect gradients
            gradient_score = self._check_gradient_perfection(image)
            
            # Check for unnatural color distributions
            color_distribution_score = self._analyze_color_histogram(image)
            
            return {
                'color_uniformity': round(uniformity_score, 2),
                'gradient_perfection': round(gradient_score, 2),
                'color_distribution_anomaly': round(color_distribution_score, 2),
                'mean_rgb': [round(m, 1) for m in mean],
                'stddev_rgb': [round(s, 1) for s in stddev]
            }
            
        except Exception as e:
            logger.error(f"Pixel analysis error: {e}")
            return {}
    
    def _analyze_frequency_domain(self, image: Any) -> Dict[str, Any]:
        """Analyze frequency domain characteristics"""
        if not self.pil_available or not np:
            return {}
            
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Apply FFT
            f_transform = np.fft.fft2(img_array)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze frequency patterns
            rows, cols = magnitude_spectrum.shape
            crow, ccol = rows // 2, cols // 2
            
            # Define regions for analysis
            center_region = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
            outer_region = magnitude_spectrum.copy()
            outer_region[crow-50:crow+50, ccol-50:ccol+50] = 0
            
            # Calculate energy in different regions
            center_energy = np.sum(center_region)
            outer_energy = np.sum(outer_region)
            total_energy = np.sum(magnitude_spectrum)
            
            # Calculate ratios
            low_freq_ratio = center_energy / total_energy if total_energy > 0 else 0
            high_freq_ratio = outer_energy / total_energy if total_energy > 0 else 0
            
            # AI images often have different frequency characteristics
            ai_frequency_score = 0
            
            # Very low high-frequency content (too smooth)
            if high_freq_ratio < 0.2:
                ai_frequency_score += 30
            
            # Unusual frequency distribution
            expected_ratio = 0.4  # Expected for natural images
            deviation = abs(low_freq_ratio - expected_ratio)
            if deviation > 0.2:
                ai_frequency_score += 20
            
            # Check for regular patterns in frequency domain
            pattern_score = self._check_frequency_patterns(magnitude_spectrum)
            ai_frequency_score += pattern_score
            
            return {
                'high_frequency_ratio': round(high_freq_ratio, 3),
                'low_frequency_ratio': round(low_freq_ratio, 3),
                'ai_frequency_score': min(ai_frequency_score, 80),
                'frequency_pattern_detected': pattern_score > 10
            }
            
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return {}
    
    def _detect_ai_artifacts(self, image: Any) -> Dict[str, Any]:
        """Detect specific AI generation artifacts"""
        artifacts = []
        
        # Check for grid patterns (DALL-E artifact)
        grid_detected, grid_strength = self._has_grid_pattern(image)
        if grid_detected:
            artifacts.append(f"Grid pattern detected (strength: {grid_strength:.1f})")
        
        # Check for unnatural edges
        edge_anomalies = self._has_unnatural_edges(image)
        if edge_anomalies:
            artifacts.extend(edge_anomalies)
        
        # Check for texture anomalies
        texture_issues = self._has_texture_anomalies(image)
        if texture_issues:
            artifacts.extend(texture_issues)
        
        return {
            'artifacts_found': artifacts,
            'artifact_count': len(artifacts)
        }
    
    def _calculate_image_ai_probability(self, metadata: Dict, pixels: Dict, 
                                      frequency: Dict, artifacts: Dict) -> float:
        """Calculate overall AI probability for image"""
        score = 0
        weights = 0
        
        # Metadata score (high weight if AI indicators found)
        if metadata.get('ai_indicators', 0) > 0:
            metadata_score = min(90, 50 + (metadata['ai_indicators'] * 20))
            score += metadata_score * 0.35
            weights += 0.35
        else:
            # Still check for suspicious lack of metadata
            if not metadata.get('has_metadata'):
                score += 30 * 0.1
                weights += 0.1
        
        # Pixel analysis score
        if pixels:
            pixel_score = 0
            
            # Color uniformity
            uniformity = pixels.get('color_uniformity', 0)
            if uniformity > 0.7:
                pixel_score += 40
            elif uniformity > 0.5:
                pixel_score += 20
            
            # Gradient perfection
            gradient = pixels.get('gradient_perfection', 0)
            if gradient > 0.8:
                pixel_score += 30
            elif gradient > 0.6:
                pixel_score += 15
            
            # Color distribution
            color_anomaly = pixels.get('color_distribution_anomaly', 0)
            if color_anomaly > 0.7:
                pixel_score += 30
            
            score += min(pixel_score, 80) * 0.25
            weights += 0.25
        
        # Frequency analysis score
        if frequency:
            freq_score = frequency.get('ai_frequency_score', 0)
            score += freq_score * 0.2
            weights += 0.2
        
        # Artifacts score
        if artifacts:
            artifact_count = artifacts.get('artifact_count', 0)
            artifact_score = min(artifact_count * 15, 80)
            score += artifact_score * 0.2
            weights += 0.2
        
        # Calculate final score
        if weights > 0:
            final_score = score / weights
        else:
            final_score = 50
        
        # Apply confidence adjustments
        if final_score > 80 and artifacts.get('artifact_count', 0) < 2:
            final_score -= 10  # Reduce confidence if few artifacts despite high score
        elif final_score < 40 and artifacts.get('artifact_count', 0) > 3:
            final_score += 10  # Increase if many artifacts despite low score
        
        return max(0, min(100, final_score))
    
    def _check_symmetry(self, image: Any) -> float:
        """Check image symmetry"""
        if not self.pil_available or not np:
            return 0
            
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            width, height = image.size
            img_array = np.array(image)
            
            # Check horizontal symmetry
            left_half = img_array[:, :width//2]
            right_half = np.fliplr(img_array[:, width//2:width//2*2])
            
            # Ensure same shape
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            if left_half.shape == right_half.shape:
                # Calculate normalized difference
                diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
                symmetry_score = 1 - (diff / 255)
                
                # Also check vertical symmetry
                top_half = img_array[:height//2, :]
                bottom_half = np.flipud(img_array[height//2:height//2*2, :])
                
                min_height = min(top_half.shape[0], bottom_half.shape[0])
                top_half = top_half[:min_height, :]
                bottom_half = bottom_half[:min_height, :]
                
                if top_half.shape == bottom_half.shape:
                    v_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
                    v_symmetry = 1 - (v_diff / 255)
                    
                    # Return maximum symmetry found
                    return max(symmetry_score, v_symmetry)
                
                return symmetry_score
            
            return 0
        except Exception as e:
            logger.error(f"Symmetry check error: {e}")
            return 0
    
    def _check_smoothness(self, image: Any) -> float:
        """Check for unrealistic smoothness"""
        if not self.pil_available or not np:
            return 0.5
            
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Calculate local variance using sliding window
            kernel_size = 5
            pad = kernel_size // 2
            
            # Use edge detection to find texture
            edges = np.array(gray.filter(ImageFilter.FIND_EDGES))
            
            # Calculate edge density
            edge_density = np.sum(edges > 30) / edges.size
            
            # Low edge density indicates smoothness
            smoothness = 1 - min(edge_density * 10, 1)
            
            # Also check variance
            local_vars = []
            step = 10  # Sample every 10 pixels for efficiency
            
            for i in range(pad, img_array.shape[0] - pad, step):
                for j in range(pad, img_array.shape[1] - pad, step):
                    window = img_array[i-pad:i+pad+1, j-pad:j+pad+1]
                    local_vars.append(np.var(window))
            
            if local_vars:
                avg_var = np.mean(local_vars)
                # Low variance indicates smoothness
                var_smoothness = 1 - min(avg_var / 500, 1)
                
                # Combine both metrics
                return (smoothness + var_smoothness) / 2
            
            return smoothness
            
        except Exception as e:
            logger.error(f"Smoothness check error: {e}")
            return 0.5
    
    def _check_repetitive_patterns(self, image: Any) -> float:
        """Check for repetitive patterns using autocorrelation"""
        if not self.pil_available or not np:
            return 0.3
            
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Downsample for efficiency
            scale = 4
            small = img_array[::scale, ::scale]
            
            # Check for repeating patterns using autocorrelation
            pattern_score = 0
            
            # Check horizontal patterns
            for row in range(0, small.shape[0], 10):
                row_data = small[row, :]
                if len(row_data) > 20:
                    # Simple autocorrelation
                    for shift in range(5, len(row_data)//2, 5):
                        if shift < len(row_data):
                            correlation = np.corrcoef(
                                row_data[:-shift], 
                                row_data[shift:]
                            )[0, 1]
                            if correlation > 0.8:
                                pattern_score += 0.1
            
            # Check vertical patterns
            for col in range(0, small.shape[1], 10):
                col_data = small[:, col]
                if len(col_data) > 20:
                    for shift in range(5, len(col_data)//2, 5):
                        if shift < len(col_data):
                            correlation = np.corrcoef(
                                col_data[:-shift], 
                                col_data[shift:]
                            )[0, 1]
                            if correlation > 0.8:
                                pattern_score += 0.1
            
            return min(pattern_score, 1.0)
            
        except Exception as e:
            logger.error(f"Pattern check error: {e}")
            return 0.3
    
    def _check_edge_artifacts(self, image: Any) -> tuple:
        """Check for AI edge artifacts"""
        if not self.pil_available:
            return [], 0
            
        try:
            artifacts = []
            score = 0
            
            # Edge detection
            edges = image.filter(ImageFilter.FIND_EDGES)
            edge_array = np.array(edges.convert('L'))
            
            # Check for unnaturally straight edges
            # Use Hough transform concept (simplified)
            height, width = edge_array.shape
            
            # Sample edge pixels
            edge_pixels = np.where(edge_array > 100)
            
            if len(edge_pixels[0]) > 100:
                # Check for perfect horizontal/vertical lines
                y_coords = edge_pixels[0]
                x_coords = edge_pixels[1]
                
                # Count perfectly aligned pixels
                h_aligned = 0
                v_aligned = 0
                
                for i in range(len(y_coords) - 1):
                    if y_coords[i] == y_coords[i + 1]:
                        h_aligned += 1
                    if x_coords[i] == x_coords[i + 1]:
                        v_aligned += 1
                
                alignment_ratio = (h_aligned + v_aligned) / len(y_coords)
                
                if alignment_ratio > 0.4:
                    artifacts.append("Unnaturally straight edges detected")
                    score += 15
                elif alignment_ratio > 0.3:
                    artifacts.append("Some artificial edge patterns")
                    score += 8
            
            # Check for edge consistency issues
            edge_variance = np.var(edge_array[edge_array > 50])
            if edge_variance < 100:
                artifacts.append("Suspiciously consistent edge strength")
                score += 10
            
            return artifacts, score
            
        except Exception as e:
            logger.error(f"Edge artifact check error: {e}")
            return [], 0
    
    def _check_gradient_perfection(self, image: Any) -> float:
        """Check for perfect gradients"""
        if not self.pil_available or not np:
            return 0.5
            
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Check multiple directions for gradients
            perfection_scores = []
            
            # Horizontal gradient check
            for row in range(0, gray.shape[0], gray.shape[0]//10):
                row_data = gray[row, :]
                if len(row_data) > 10:
                    # Calculate differences
                    diffs = np.diff(row_data)
                    if len(diffs) > 0 and np.std(diffs) > 0:
                        # Check how consistent the differences are
                        consistency = 1 - (np.std(diffs) / (np.mean(np.abs(diffs)) + 1))
                        perfection_scores.append(consistency)
            
            # Vertical gradient check
            for col in range(0, gray.shape[1], gray.shape[1]//10):
                col_data = gray[:, col]
                if len(col_data) > 10:
                    diffs = np.diff(col_data)
                    if len(diffs) > 0 and np.std(diffs) > 0:
                        consistency = 1 - (np.std(diffs) / (np.mean(np.abs(diffs)) + 1))
                        perfection_scores.append(consistency)
            
            if perfection_scores:
                # Return the maximum perfection found
                return min(max(perfection_scores), 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Gradient check error: {e}")
            return 0.5
    
    def _has_grid_pattern(self, image: Any) -> tuple:
        """Check for grid patterns (common in DALL-E)"""
        if not self.pil_available or not np:
            return False, 0
            
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Edge detection to find lines
            edges = image.filter(ImageFilter.FIND_EDGES)
            edge_array = np.array(edges.convert('L'))
            
            # Look for regular spacing in edges
            # Sum along axes to find peaks
            h_projection = np.sum(edge_array > 100, axis=1)
            v_projection = np.sum(edge_array > 100, axis=0)
            
            # Find peaks (potential grid lines)
            h_peaks = self._find_regular_peaks(h_projection)
            v_peaks = self._find_regular_peaks(v_projection)
            
            # If we find regular peaks in both directions, it's likely a grid
            if len(h_peaks) > 3 and len(v_peaks) > 3:
                # Calculate regularity score
                h_regularity = self._calculate_spacing_regularity(h_peaks)
                v_regularity = self._calculate_spacing_regularity(v_peaks)
                
                grid_strength = (h_regularity + v_regularity) / 2
                
                if grid_strength > 0.7:
                    return True, grid_strength
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Grid pattern check error: {e}")
            return False, 0
    
    def _has_unnatural_edges(self, image: Any) -> List[str]:
        """Check for unnatural edges"""
        if not self.pil_available:
            return []
            
        try:
            anomalies = []
            
            # Get edges
            edges = image.filter(ImageFilter.FIND_EDGES)
            edge_array = np.array(edges.convert('L'))
            
            # Check for edges that are too perfect
            strong_edges = edge_array > 150
            
            if np.sum(strong_edges) > 100:
                # Analyze edge characteristics
                edge_widths = []
                
                # Sample random edge points and measure width
                edge_points = np.where(strong_edges)
                if len(edge_points[0]) > 50:
                    sample_indices = np.random.choice(
                        len(edge_points[0]), 
                        min(50, len(edge_points[0])), 
                        replace=False
                    )
                    
                    for idx in sample_indices:
                        y, x = edge_points[0][idx], edge_points[1][idx]
                        
                        # Measure edge width (simplified)
                        width = 1
                        for d in range(1, 10):
                            if (x + d < edge_array.shape[1] and 
                                edge_array[y, x + d] > 100):
                                width += 1
                            else:
                                break
                        
                        edge_widths.append(width)
                    
                    if edge_widths:
                        avg_width = np.mean(edge_widths)
                        width_variance = np.var(edge_widths)
                        
                        # Unnatural if edges are too consistent
                        if width_variance < 0.5 and avg_width > 1:
                            anomalies.append("Unnaturally consistent edge width")
                        
                        if avg_width > 5:
                            anomalies.append("Unusually thick edges detected")
            
            # Check for impossible edge transitions
            # Look for edges that appear/disappear suddenly
            edge_continuity = self._check_edge_continuity(edge_array)
            if edge_continuity < 0.6:
                anomalies.append("Discontinuous edges detected")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Edge analysis error: {e}")
            return []
    
    def _has_texture_anomalies(self, image: Any) -> List[str]:
        """Check for texture anomalies"""
        if not self.pil_available or not np:
            return []
            
        try:
            anomalies = []
            
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Analyze texture using local binary patterns (simplified)
            # Check for areas that are too uniform
            block_size = 32
            uniformity_scores = []
            
            for y in range(0, gray.shape[0] - block_size, block_size):
                for x in range(0, gray.shape[1] - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    
                    # Calculate local statistics
                    block_std = np.std(block)
                    block_range = np.max(block) - np.min(block)
                    
                    # Low variance and range indicate uniform texture
                    if block_std < 5 and block_range < 20:
                        uniformity_scores.append(1)
                    else:
                        uniformity_scores.append(0)
            
            if uniformity_scores:
                uniformity_ratio = sum(uniformity_scores) / len(uniformity_scores)
                
                if uniformity_ratio > 0.3:
                    anomalies.append(f"Large uniform texture areas ({uniformity_ratio*100:.0f}% of image)")
                
                # Check for repeating texture patterns
                if self._has_repeating_textures(gray):
                    anomalies.append("Repeating texture patterns detected")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Texture analysis error: {e}")
            return []
    
    def _check_model_patterns(self, image: Any, model: str, patterns: List[str]) -> float:
        """Check for model-specific patterns"""
        score = 0
        
        try:
            # Model-specific checks
            if model == 'midjourney':
                # MidJourney often has very high quality and specific style
                if self._check_midjourney_style(image):
                    score += 30
                    
            elif model == 'dalle':
                # DALL-E can have grid artifacts and specific color handling
                grid_detected, _ = self._has_grid_pattern(image)
                if grid_detected:
                    score += 25
                    
            elif model == 'stable_diffusion':
                # SD often has specific noise patterns
                if self._check_sd_noise_pattern(image):
                    score += 25
            
            # Generic pattern checks
            if 'perfect symmetry' in patterns:
                symmetry = self._check_symmetry(image)
                if symmetry > 0.9:
                    score += 15
                    
            if 'unrealistic lighting' in patterns:
                if self._check_lighting_consistency(image):
                    score += 10
                    
            if 'texture artifacts' in patterns:
                if self._has_texture_anomalies(image):
                    score += 10
            
            return min(score, 50)  # Cap at 50 to require metadata for high confidence
            
        except Exception as e:
            logger.error(f"Model pattern check error: {e}")
            return 0
    
    def _error_level_analysis(self, image: Any) -> Dict[str, Any]:
        """Perform error level analysis"""
        try:
            anomalies = []
            
            # Save at different quality levels and compare
            buffer_high = BytesIO()
            buffer_low = BytesIO()
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save at different qualities
            image.save(buffer_high, format='JPEG', quality=95)
            image.save(buffer_low, format='JPEG', quality=90)
            
            # Reload and compare
            buffer_high.seek(0)
            buffer_low.seek(0)
            
            img_high = Image.open(buffer_high)
            img_low = Image.open(buffer_low)
            
            # Calculate difference
            diff = ImageChops.difference(img_high, img_low)
            diff_array = np.array(diff)
            
            # Analyze difference patterns
            mean_diff = np.mean(diff_array)
            std_diff = np.std(diff_array)
            
            # Check for uniform compression (sign of AI generation)
            if std_diff < 5:
                anomalies.append("Uniform compression artifacts detected")
            
            # Check for areas with no compression artifacts (too perfect)
            flat_areas = np.sum(diff_array < 2) / diff_array.size
            if flat_areas > 0.3:
                anomalies.append("Suspiciously perfect areas with no compression artifacts")
            
            return {
                'anomalies': anomalies,
                'mean_error_level': float(mean_diff),
                'error_level_variance': float(std_diff)
            }
            
        except Exception as e:
            logger.error(f"ELA error: {e}")
            return {'anomalies': []}
    
    def _analyze_compression_artifacts(self, image: Any) -> Dict[str, Any]:
        """Analyze compression artifacts"""
        try:
            artifacts = []
            
            # Check for JPEG block artifacts
            if image.format == 'JPEG' or image.mode == 'RGB':
                # Look for 8x8 block boundaries
                gray = np.array(image.convert('L'))
                
                # Calculate differences at block boundaries
                block_diffs = []
                for y in range(7, gray.shape[0] - 8, 8):
                    for x in range(7, gray.shape[1] - 8, 8):
                        # Check boundary differences
                        h_diff = abs(int(gray[y, x]) - int(gray[y + 1, x]))
                        v_diff = abs(int(gray[y, x]) - int(gray[y, x + 1]))
                        block_diffs.append(max(h_diff, v_diff))
                
                if block_diffs:
                    avg_block_diff = np.mean(block_diffs)
                    
                    # AI images might have unusual block artifact patterns
                    if avg_block_diff < 3:
                        artifacts.append("Unusually low JPEG block artifacts")
                    elif avg_block_diff > 50:
                        artifacts.append("Excessive block artifacts")
            
            return {'ai_artifacts': artifacts}
            
        except Exception as e:
            logger.error(f"Compression analysis error: {e}")
            return {'ai_artifacts': []}
    
    def _analyze_color_distribution(self, image: Any) -> Dict[str, Any]:
        """Analyze color distribution"""
        try:
            anomalies = []
            
            # Get color histogram
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Analyze each channel
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                hist = image.histogram()[i*256:(i+1)*256]
                
                # Check for unnatural spikes or gaps
                zero_bins = sum(1 for h in hist if h == 0)
                max_bin = max(hist)
                avg_bin = sum(hist) / len(hist)
                
                # Too many empty bins might indicate artificial generation
                if zero_bins > 100:
                    anomalies.append(f"{channel} channel has unnatural gaps")
                
                # Check for unnatural peaks
                if max_bin > avg_bin * 50:
                    anomalies.append(f"{channel} channel has unnatural color concentration")
            
            # Check overall color count
            unique_colors = len(set(image.getdata()))
            total_pixels = image.size[0] * image.size[1]
            color_ratio = unique_colors / total_pixels
            
            if color_ratio < 0.1:
                anomalies.append("Unusually low color diversity")
            
            return {'anomalies': anomalies}
            
        except Exception as e:
            logger.error(f"Color distribution error: {e}")
            return {'anomalies': []}
    
    def _analyze_edge_coherence(self, image: Any) -> Dict[str, Any]:
        """Analyze edge coherence"""
        try:
            inconsistencies = []
            
            # Get edges at different scales
            edges1 = np.array(image.filter(ImageFilter.FIND_EDGES).convert('L'))
            
            # Resize and get edges again
            small = image.resize((image.size[0]//2, image.size[1]//2), Image.Resampling.LANCZOS)
            edges2 = np.array(small.filter(ImageFilter.FIND_EDGES).convert('L'))
            edges2_resized = np.array(Image.fromarray(edges2).resize(image.size, Image.Resampling.LANCZOS))
            
            # Compare edge maps
            if edges1.shape == edges2_resized.shape:
                difference = np.abs(edges1.astype(float) - edges2_resized.astype(float))
                avg_diff = np.mean(difference)
                
                # High difference indicates scale-dependent artifacts
                if avg_diff > 50:
                    inconsistencies.append("Scale-dependent edge artifacts")
            
            return {'inconsistencies': inconsistencies}
            
        except Exception as e:
            logger.error(f"Edge coherence error: {e}")
            return {'inconsistencies': []}
    
    def _calculate_forensic_probability(self, ela: Dict, compression: Dict, 
                                      color: Dict, edge: Dict) -> float:
        """Calculate probability based on forensic analysis"""
        score = 30  # Base score
        
        # Add points for each anomaly type
        score += len(ela.get('anomalies', [])) * 15
        score += len(compression.get('ai_artifacts', [])) * 12
        score += len(color.get('anomalies', [])) * 8
        score += len(edge.get('inconsistencies', [])) * 10
        
        # Additional scoring based on specific findings
        if ela.get('mean_error_level', 10) < 3:
            score += 15  # Very low error levels are suspicious
        
        if ela.get('error_level_variance', 10) < 2:
            score += 10  # Too uniform
        
        return min(score, 95)
    
    def _create_image_summary(self, ai_probability: float, artifacts: Dict) -> str:
        """Create summary for image analysis"""
        artifact_count = artifacts.get('artifact_count', 0)
        
        if ai_probability >= 80:
            return f"This image shows strong signs of AI generation with {artifact_count} artifacts detected. High probability of being created by AI image generation tools."
        elif ai_probability >= 60:
            return f"This image likely contains AI-generated elements. {artifact_count} suspicious patterns were found that are common in AI-generated images."
        elif ai_probability >= 40:
            return f"Mixed indicators present with {artifact_count} potential artifacts. The image may have some AI involvement or heavy digital processing."
        else:
            return f"This image appears to be authentic with minimal signs of AI generation. Only {artifact_count} minor anomalies detected."
    
    def _detect_model(self, metadata: Dict, artifacts: Dict, pixels: Dict) -> str:
        """Detect which AI model might have generated the image"""
        # Check for software in metadata first
        if metadata.get('detected_software'):
            software = metadata['detected_software'][0]
            
            if 'midjourney' in software:
                return 'Midjourney'
            elif 'dalle' in software or 'dall-e' in software:
                return 'DALL-E'
            elif 'stable diffusion' in software or 'automatic1111' in software:
                return 'Stable Diffusion'
            elif 'leonardo' in software:
                return 'Leonardo AI'
            elif 'firefly' in software:
                return 'Adobe Firefly'
        
        # If no metadata, try to infer from characteristics
        if artifacts.get('artifacts_found'):
            artifacts_str = ' '.join(artifacts['artifacts_found'])
            
            if 'grid pattern' in artifacts_str.lower():
                return 'DALL-E (suspected)'
            elif pixels and pixels.get('gradient_perfection', 0) > 0.8:
                return 'Midjourney (suspected)'
        
        if metadata.get('ai_indicators', 0) > 0:
            return 'Unknown AI Model'
        
        return 'Not Detected'
    
    # Additional helper methods
    
    def _check_color_banding(self, image: Any) -> float:
        """Check for color banding artifacts"""
        if not self.pil_available or not np:
            return 0
            
        try:
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check each color channel
            banding_scores = []
            
            for channel in range(3):
                channel_data = np.array(image)[:, :, channel]
                
                # Look for steps in what should be gradients
                # Sample random rows/columns
                for _ in range(10):
                    if np.random.rand() > 0.5:
                        # Sample row
                        row = np.random.randint(0, channel_data.shape[0])
                        data = channel_data[row, :]
                    else:
                        # Sample column  
                        col = np.random.randint(0, channel_data.shape[1])
                        data = channel_data[:, col]
                    
                    if len(data) > 10:
                        # Calculate consecutive differences
                        diffs = np.diff(data)
                        
                        # Count sudden changes
                        sudden_changes = np.sum(np.abs(diffs) > 10)
                        
                        # In smooth gradients, we shouldn't see many sudden changes
                        if sudden_changes > len(diffs) * 0.1:
                            banding_scores.append(1)
                        else:
                            banding_scores.append(0)
            
            return sum(banding_scores) / len(banding_scores) if banding_scores else 0
            
        except Exception as e:
            logger.error(f"Color banding check error: {e}")
            return 0
    
    def _check_noise_patterns(self, image: Any) -> float:
        """Check for artificial noise patterns"""
        if not self.pil_available or not np:
            return 0
            
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Calculate local noise levels
            noise_map = np.zeros_like(gray, dtype=float)
            
            # Use a simple high-pass filter to detect noise
            kernel_size = 3
            for i in range(kernel_size//2, gray.shape[0] - kernel_size//2):
                for j in range(kernel_size//2, gray.shape[1] - kernel_size//2):
                    local = gray[i-1:i+2, j-1:j+2]
                    noise_map[i, j] = np.std(local)
            
            # Analyze noise distribution
            noise_flat = noise_map.flatten()
            noise_flat = noise_flat[noise_flat > 0]
            
            if len(noise_flat) > 100:
                # Check if noise is too uniform (artificial)
                noise_variance = np.var(noise_flat)
                mean_noise = np.mean(noise_flat)
                
                # Artificial noise tends to be very uniform
                if mean_noise > 5 and noise_variance < mean_noise * 0.5:
                    return 0.8
                elif mean_noise > 3 and noise_variance < mean_noise:
                    return 0.5
            
            return 0.2
            
        except Exception as e:
            logger.error(f"Noise pattern check error: {e}")
            return 0
    
    def _find_regular_peaks(self, projection: np.ndarray, min_distance: int = 10) -> List[int]:
        """Find regularly spaced peaks in projection"""
        peaks = []
        threshold = np.mean(projection) + np.std(projection)
        
        i = 0
        while i < len(projection):
            if projection[i] > threshold:
                # Find local maximum
                local_max = i
                while i < len(projection) and projection[i] > threshold:
                    if projection[i] > projection[local_max]:
                        local_max = i
                    i += 1
                peaks.append(local_max)
                i += min_distance
            else:
                i += 1
        
        return peaks
    
    def _calculate_spacing_regularity(self, peaks: List[int]) -> float:
        """Calculate how regular the spacing between peaks is"""
        if len(peaks) < 3:
            return 0
        
        spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        
        if not spacings:
            return 0
        
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        # Lower std relative to mean indicates regularity
        if mean_spacing > 0:
            regularity = 1 - min(std_spacing / mean_spacing, 1)
            return regularity
        
        return 0
    
    def _check_edge_continuity(self, edge_array: np.ndarray) -> float:
        """Check edge continuity score"""
        # Simplified continuity check
        # Real implementation would trace edges and check for breaks
        
        # For now, check if edges form connected components
        strong_edges = edge_array > 100
        edge_density = np.sum(strong_edges) / strong_edges.size
        
        # Higher density usually means more continuous edges
        if edge_density > 0.1:
            return 0.8
        elif edge_density > 0.05:
            return 0.6
        else:
            return 0.4
    
    def _has_repeating_textures(self, gray_array: np.ndarray) -> bool:
        """Check for repeating textures using template matching"""
        try:
            # Sample a small patch
            patch_size = 32
            h, w = gray_array.shape
            
            if h < patch_size * 3 or w < patch_size * 3:
                return False
            
            # Take a patch from the center
            center_y, center_x = h // 2, w // 2
            patch = gray_array[
                center_y - patch_size//2:center_y + patch_size//2,
                center_x - patch_size//2:center_x + patch_size//2
            ]
            
            # Look for this patch elsewhere
            matches = 0
            
            # Sample a few locations
            for _ in range(10):
                y = np.random.randint(patch_size, h - patch_size)
                x = np.random.randint(patch_size, w - patch_size)
                
                test_patch = gray_array[y:y+patch_size, x:x+patch_size]
                
                if test_patch.shape == patch.shape:
                    # Calculate similarity
                    diff = np.mean(np.abs(patch.astype(float) - test_patch.astype(float)))
                    
                    if diff < 10:  # Very similar
                        matches += 1
            
            return matches > 2
            
        except Exception as e:
            logger.error(f"Texture repetition check error: {e}")
            return False
    
    def _check_frequency_patterns(self, magnitude_spectrum: np.ndarray) -> float:
        """Check for regular patterns in frequency domain"""
        try:
            # Look for regular patterns that shouldn't exist in natural images
            score = 0
            
            # Check for strong regular frequencies
            # Flatten and sort to find peaks
            flat_spectrum = magnitude_spectrum.flatten()
            sorted_spectrum = np.sort(flat_spectrum)[::-1]
            
            # Get top frequencies
            top_freqs = sorted_spectrum[:100]
            
            # Check if top frequencies are too regular
            if len(top_freqs) > 10:
                freq_diffs = np.diff(top_freqs)
                
                # Regular spacing indicates artificial patterns
                if np.std(freq_diffs) < np.mean(np.abs(freq_diffs)) * 0.3:
                    score += 20
            
            return score
            
        except Exception as e:
            logger.error(f"Frequency pattern check error: {e}")
            return 0
    
    def _analyze_color_histogram(self, image: Any) -> float:
        """Analyze color histogram for anomalies"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            anomaly_score = 0
            
            # Get histograms for each channel
            hist = image.histogram()
            
            for i in range(3):  # R, G, B
                channel_hist = hist[i*256:(i+1)*256]
                
                # Check for unnatural patterns
                # 1. Too many zero bins (gaps in histogram)
                zero_bins = sum(1 for h in channel_hist if h == 0)
                if zero_bins > 150:
                    anomaly_score += 0.2
                
                # 2. Unnatural spikes
                mean_count = sum(channel_hist) / 256
                spike_count = sum(1 for h in channel_hist if h > mean_count * 10)
                if spike_count > 5:
                    anomaly_score += 0.2
                
                # 3. Too smooth histogram (over-processed)
                hist_diffs = [abs(channel_hist[i] - channel_hist[i+1]) 
                             for i in range(255)]
                if sum(hist_diffs) < mean_count * 50:
                    anomaly_score += 0.1
            
            return min(anomaly_score, 1.0)
            
        except Exception as e:
            logger.error(f"Histogram analysis error: {e}")
            return 0
    
    def _check_midjourney_style(self, image: Any) -> bool:
        """Check for Midjourney-specific characteristics"""
        try:
            # Midjourney tends to have:
            # 1. Very high detail and sharpness
            # 2. Specific color grading
            # 3. Often symmetrical compositions
            
            # Check sharpness
            edges = np.array(image.filter(ImageFilter.FIND_EDGES).convert('L'))
            edge_strength = np.mean(edges[edges > 50])
            
            # Check color characteristics
            if image.mode == 'RGB':
                img_array = np.array(image)
                
                # Midjourney often has rich, saturated colors
                saturation = self._calculate_saturation(img_array)
                
                if edge_strength > 100 and saturation > 0.6:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Midjourney check error: {e}")
            return False
    
    def _check_sd_noise_pattern(self, image: Any) -> bool:
        """Check for Stable Diffusion noise patterns"""
        try:
            # SD often has characteristic noise in certain frequency ranges
            gray = np.array(image.convert('L'))
            
            # Check high-frequency noise
            noise = gray - np.array(image.convert('L').filter(ImageFilter.GaussianBlur(2)))
            
            # SD noise has specific statistical properties
            noise_std = np.std(noise)
            noise_mean = np.abs(np.mean(noise))
            
            # SD typically has noise_std between 3-8 with near-zero mean
            if 3 < noise_std < 8 and noise_mean < 1:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"SD noise check error: {e}")
            return False
    
    def _check_lighting_consistency(self, image: Any) -> bool:
        """Check if lighting is unnaturally consistent"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to LAB color space (approximation)
            # This is simplified - real LAB conversion is more complex
            img_array = np.array(image)
            
            # Simple luminance calculation
            luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            
            # Divide image into regions and check lighting consistency
            h, w = luminance.shape
            region_size = min(h, w) // 4
            
            region_means = []
            
            for i in range(0, h - region_size, region_size):
                for j in range(0, w - region_size, region_size):
                    region = luminance[i:i+region_size, j:j+region_size]
                    region_means.append(np.mean(region))
            
            if region_means:
                # Check if lighting is too consistent across regions
                lighting_variance = np.var(region_means)
                
                # Unnatural if variance is very low (flat lighting)
                if lighting_variance < 100:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Lighting check error: {e}")
            return False
    
    def _calculate_saturation(self, rgb_array: np.ndarray) -> float:
        """Calculate average saturation of image"""
        try:
            # Convert RGB to HSV (simplified)
            r, g, b = rgb_array[:,:,0]/255.0, rgb_array[:,:,1]/255.0, rgb_array[:,:,2]/255.0
            
            max_rgb = np.maximum(np.maximum(r, g), b)
            min_rgb = np.minimum(np.minimum(r, g), b)
            
            diff = max_rgb - min_rgb
            
            # Saturation is (max - min) / max
            saturation = np.where(max_rgb > 0, diff / max_rgb, 0)
            
            return np.mean(saturation)
            
        except Exception as e:
            logger.error(f"Saturation calculation error: {e}")
            return 0.5
    
    def _basic_analysis(self, image_data: str) -> Dict[str, Any]:
        """Basic analysis when PIL is not available"""
        # Simple heuristic based on image data size and patterns
        data_size = len(image_data)
        
        # Check for common AI image data patterns
        ai_probability = 50
        
        # Very large images might be AI (high resolution)
        if data_size > 1000000:  # > 1MB base64
            ai_probability += 10
        
        # Check for repetitive base64 patterns (simplified)
        sample_size = min(1000, data_size // 10)
        if sample_size > 100:
            # Sample different parts of the data
            samples = []
            for i in range(5):
                start = (data_size // 6) * i
                samples.append(image_data[start:start+20])
            
            # Check for unusual similarity
            unique_samples = len(set(samples))
            if unique_samples < 3:
                ai_probability += 20
        
        return {
            'ai_probability': ai_probability,
            'note': 'Limited analysis available. Install Pillow for full image analysis.',
            'data_size': data_size,
            'metadata_analysis': {'has_metadata': False, 'ai_indicators': 0},
            'pixel_analysis': {},
            'frequency_analysis': {},
            'artifact_analysis': {'artifacts_found': [], 'artifact_count': 0}
        }
    
    def _summarize_artifacts(self, ela: Dict, compression: Dict, 
                           color: Dict, edge: Dict) -> List[str]:
        """Summarize all detected artifacts"""
        artifacts = []
        
        # Add all artifacts with priority
        if ela.get('anomalies'):
            artifacts.extend([f"ELA: {a}" for a in ela['anomalies'][:2]])
        
        if compression.get('ai_artifacts'):
            artifacts.extend([f"Compression: {a}" for a in compression['ai_artifacts'][:2]])
        
        if color.get('anomalies'):
            artifacts.extend([f"Color: {a}" for a in color['anomalies'][:2]])
        
        if edge.get('inconsistencies'):
            artifacts.extend([f"Edge: {a}" for a in edge['inconsistencies'][:2]])
        
        return artifacts[:5]  # Return top 5
