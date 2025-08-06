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
                # More intelligent basic analysis
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
            
            # Calculate overall AI probability with better calibration
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
                return {'ai_score': 75}  # Higher default for missing PIL
            
            image = self._decode_image(image_data)
            if not image:
                return {'ai_score': 75}
            
            artifacts = []
            ai_score = 30  # Start with higher base score
            
            # Check for perfect symmetry (common in AI)
            symmetry_score = self._check_symmetry(image)
            if symmetry_score > 0.8:
                artifacts.append("Perfect symmetry detected")
                ai_score += 30
            elif symmetry_score > 0.6:
                artifacts.append("High symmetry detected")
                ai_score += 20
            elif symmetry_score > 0.4:
                artifacts.append("Moderate symmetry detected")
                ai_score += 10
            
            # Check for unrealistic smoothness
            smoothness = self._check_smoothness(image)
            if smoothness > 0.7:
                artifacts.append("Unrealistic surface smoothness")
                ai_score += 25
            elif smoothness > 0.5:
                artifacts.append("Unusual smoothness detected")
                ai_score += 15
            elif smoothness > 0.3:
                artifacts.append("Some smooth areas detected")
                ai_score += 8
            
            # Check for repetitive patterns
            repetition = self._check_repetitive_patterns(image)
            if repetition > 0.5:
                artifacts.append("Repetitive pattern artifacts")
                ai_score += 25
            elif repetition > 0.3:
                artifacts.append("Some pattern repetition detected")
                ai_score += 15
            elif repetition > 0.2:
                artifacts.append("Minor repetition detected")
                ai_score += 8
            
            # Check for edge artifacts
            edge_artifacts, edge_score = self._check_edge_artifacts(image)
            if edge_artifacts:
                artifacts.extend(edge_artifacts)
                ai_score += edge_score
            
            # Check for color banding
            banding_score = self._check_color_banding(image)
            if banding_score > 0.4:
                artifacts.append("Color banding detected")
                ai_score += 20
            elif banding_score > 0.2:
                artifacts.append("Minor color banding")
                ai_score += 10
            
            # Check for noise patterns
            noise_score = self._check_noise_patterns(image)
            if noise_score > 0.5:
                artifacts.append("Artificial noise patterns")
                ai_score += 20
            elif noise_score > 0.3:
                artifacts.append("Unusual noise distribution")
                ai_score += 10
            
            # Check for too-perfect aspects
            perfection_score = self._check_perfection_indicators(image)
            if perfection_score > 0.7:
                artifacts.append("Unnaturally perfect elements")
                ai_score += 25
            elif perfection_score > 0.5:
                artifacts.append("Suspiciously clean rendering")
                ai_score += 15
            
            return {
                'ai_score': min(ai_score, 95),
                'artifacts': artifacts,
                'artifact_count': len(artifacts)
            }
            
        except Exception as e:
            logger.error(f"Artifact detection error: {str(e)}", exc_info=True)
            return {'ai_score': 75, 'error': str(e)}
    
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
                'comfyui', 'leonardo', 'firefly', 'runway', 'wombo', 'nightcafe',
                'playground', 'lexica', 'craiyon', 'starryai', 'hotpot'
            ]
            
            for keyword in ai_keywords:
                if keyword in metadata_str:
                    ai_indicators += 3  # Increased weight
                    detected_software.append(keyword)
            
            # Check EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                try:
                    exif = image._getexif()
                    # Check for AI software in EXIF
                    software_tag = 0x0131  # Software EXIF tag
                    if software_tag in exif:
                        software = str(exif[software_tag]).lower()
                        for keyword in ai_keywords:
                            if keyword in software:
                                ai_indicators += 4  # Higher weight for EXIF
                                detected_software.append(f"EXIF: {keyword}")
                except:
                    pass
        
        # Modern AI images often lack traditional metadata
        if not metadata or len(metadata) < 3:
            ai_indicators += 2  # Suspicious lack of metadata
        
        return {
            'has_metadata': bool(metadata),
            'ai_indicators': ai_indicators,
            'suspicious_metadata': ai_indicators > 0,
            'detected_software': list(set(detected_software)),
            'metadata_fields': len(metadata) if metadata else 0
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
                if std < 10:  # Very low standard deviation
                    uniformity_score += 0.5
                elif std < 20:
                    uniformity_score += 0.35
                elif std < 30:
                    uniformity_score += 0.2
            
            # Check for perfect gradients
            gradient_score = self._check_gradient_perfection(image)
            
            # Check for unnatural color distributions
            color_distribution_score = self._analyze_color_histogram(image)
            
            # AI images often have very smooth areas
            if uniformity_score > 0.4:
                uniformity_score *= 1.3  # Boost score for uniformity
            
            # Check for telltale AI color patterns
            ai_color_score = self._check_ai_color_patterns(image)
            
            return {
                'color_uniformity': round(uniformity_score, 2),
                'gradient_perfection': round(gradient_score, 2),
                'color_distribution_anomaly': round(color_distribution_score, 2),
                'ai_color_patterns': round(ai_color_score, 2),
                'mean_rgb': [round(m, 1) for m in mean],
                'stddev_rgb': [round(s, 1) for s in stddev]
            }
            
        except Exception as e:
            logger.error(f"Pixel analysis error: {e}")
            return {}
    
    def _analyze_frequency_domain(self, image: Any) -> Dict[str, Any]:
        """Analyze frequency domain characteristics"""
        if not self.pil_available or not np:
            return {'ai_frequency_score': 60}  # Default higher score
            
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
            center_size = min(30, rows//4, cols//4)
            center_region = magnitude_spectrum[crow-center_size:crow+center_size, 
                                             ccol-center_size:ccol+center_size]
            
            # Create outer region mask
            outer_region = magnitude_spectrum.copy()
            outer_size = min(50, rows//3, cols//3)
            outer_region[crow-outer_size:crow+outer_size, ccol-outer_size:ccol+outer_size] = 0
            
            # Calculate energy in different regions
            center_energy = np.sum(center_region)
            outer_energy = np.sum(outer_region)
            total_energy = np.sum(magnitude_spectrum)
            
            # Calculate ratios
            low_freq_ratio = center_energy / total_energy if total_energy > 0 else 0
            high_freq_ratio = outer_energy / total_energy if total_energy > 0 else 0
            
            # AI images often have different frequency characteristics
            ai_frequency_score = 40  # Start with higher base
            
            # Very low high-frequency content (too smooth)
            if high_freq_ratio < 0.1:
                ai_frequency_score += 40
            elif high_freq_ratio < 0.2:
                ai_frequency_score += 25
            elif high_freq_ratio < 0.3:
                ai_frequency_score += 15
            
            # Unusual frequency distribution
            expected_ratio = 0.35  # Expected for natural images
            deviation = abs(low_freq_ratio - expected_ratio)
            if deviation > 0.2:
                ai_frequency_score += 25
            elif deviation > 0.15:
                ai_frequency_score += 15
            elif deviation > 0.1:
                ai_frequency_score += 10
            
            # Check for regular patterns in frequency domain
            pattern_score = self._check_frequency_patterns(magnitude_spectrum)
            ai_frequency_score += pattern_score
            
            return {
                'high_frequency_ratio': round(high_freq_ratio, 3),
                'low_frequency_ratio': round(low_freq_ratio, 3),
                'ai_frequency_score': min(ai_frequency_score, 95),
                'frequency_pattern_detected': pattern_score > 10
            }
            
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return {'ai_frequency_score': 60}
    
    def _detect_ai_artifacts(self, image: Any) -> Dict[str, Any]:
        """Detect specific AI generation artifacts"""
        artifacts = []
        
        # Check for grid patterns (DALL-E artifact)
        grid_detected, grid_strength = self._has_grid_pattern(image)
        if grid_detected:
            artifacts.append(f"Grid pattern detected (strength: {grid_strength:.1f})")
        elif grid_strength > 0.3:
            artifacts.append("Subtle grid-like structures")
        
        # Check for unnatural edges
        edge_anomalies = self._has_unnatural_edges(image)
        if edge_anomalies:
            artifacts.extend(edge_anomalies)
        
        # Check for texture anomalies
        texture_issues = self._has_texture_anomalies(image)
        if texture_issues:
            artifacts.extend(texture_issues)
        
        # Check for AI-specific rendering artifacts
        rendering_artifacts = self._check_ai_rendering_artifacts(image)
        if rendering_artifacts:
            artifacts.extend(rendering_artifacts)
        
        return {
            'artifacts_found': artifacts,
            'artifact_count': len(artifacts)
        }
    
    def _calculate_image_ai_probability(self, metadata: Dict, pixels: Dict, 
                                      frequency: Dict, artifacts: Dict) -> float:
        """Calculate overall AI probability for image"""
        # Start with base probability that assumes modern AI
        base_score = 45  # Much higher base assumption
        
        score = 0
        weights = 0
        
        # Metadata score (high weight if AI indicators found)
        if metadata.get('ai_indicators', 0) > 0:
            # Strong evidence in metadata
            metadata_score = min(95, 70 + (metadata['ai_indicators'] * 5))
            score += metadata_score * 0.3
            weights += 0.3
        else:
            # Lack of metadata is suspicious for modern images
            if not metadata.get('has_metadata') or metadata.get('metadata_fields', 10) < 3:
                score += 60 * 0.15  # Suspicious
                weights += 0.15
            else:
                score += 20 * 0.1
                weights += 0.1
        
        # Pixel analysis score
        if pixels:
            pixel_score = 0
            
            # Color uniformity
            uniformity = pixels.get('color_uniformity', 0)
            if uniformity > 0.7:
                pixel_score += 50
            elif uniformity > 0.5:
                pixel_score += 35
            elif uniformity > 0.3:
                pixel_score += 20
            else:
                pixel_score += 10
            
            # Gradient perfection
            gradient = pixels.get('gradient_perfection', 0)
            if gradient > 0.6:
                pixel_score += 40
            elif gradient > 0.4:
                pixel_score += 25
            elif gradient > 0.2:
                pixel_score += 15
            
            # Color distribution
            color_anomaly = pixels.get('color_distribution_anomaly', 0)
            if color_anomaly > 0.5:
                pixel_score += 35
            elif color_anomaly > 0.3:
                pixel_score += 20
            elif color_anomaly > 0.2:
                pixel_score += 10
            
            # AI color patterns
            ai_colors = pixels.get('ai_color_patterns', 0)
            if ai_colors > 0.6:
                pixel_score += 30
            elif ai_colors > 0.4:
                pixel_score += 20
            elif ai_colors > 0.2:
                pixel_score += 10
            
            score += min(pixel_score, 90) * 0.25
            weights += 0.25
        
        # Frequency analysis score
        if frequency:
            freq_score = frequency.get('ai_frequency_score', 50)
            score += min(freq_score, 90) * 0.25
            weights += 0.25
        
        # Artifacts score
        if artifacts:
            artifact_count = artifacts.get('artifact_count', 0)
            if artifact_count >= 4:
                artifact_score = min(artifact_count * 18, 90)
            elif artifact_count >= 2:
                artifact_score = min(artifact_count * 22, 80)
            else:
                artifact_score = min(artifact_count * 25, 60)
            score += artifact_score * 0.2
            weights += 0.2
        
        # Calculate weighted score
        if weights > 0:
            weighted_score = score / weights
        else:
            weighted_score = 0
        
        # Combine with base score
        final_score = base_score * 0.3 + weighted_score * 0.7
        
        # Boost score if multiple strong indicators
        strong_indicators = 0
        if metadata.get('ai_indicators', 0) > 2:
            strong_indicators += 1
        if pixels and pixels.get('color_uniformity', 0) > 0.5:
            strong_indicators += 1
        if pixels and pixels.get('gradient_perfection', 0) > 0.5:
            strong_indicators += 1
        if frequency and frequency.get('ai_frequency_score', 0) > 60:
            strong_indicators += 1
        if artifacts and artifacts.get('artifact_count', 0) > 2:
            strong_indicators += 1
        
        # Multiple indicators strongly suggest AI
        if strong_indicators >= 4:
            final_score = max(final_score, 85)
        elif strong_indicators >= 3:
            final_score = max(final_score, 75)
        elif strong_indicators >= 2:
            final_score = max(final_score, 65)
        
        # Apply minimum threshold for any detected indicators
        if (metadata.get('ai_indicators', 0) > 0 or 
            artifacts.get('artifact_count', 0) > 0 or
            (pixels and pixels.get('ai_color_patterns', 0) > 0.3)):
            final_score = max(final_score, 55)
        
        return max(0, min(100, final_score))
    
    def _check_symmetry(self, image: Any) -> float:
        """Check image symmetry"""
        if not self.pil_available or not np:
            return 0.3  # Default moderate value
            
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
            
            return 0.3
        except Exception as e:
            logger.error(f"Symmetry check error: {e}")
            return 0.3
    
    def _check_smoothness(self, image: Any) -> float:
        """Check for unrealistic smoothness"""
        if not self.pil_available or not np:
            return 0.5
            
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Use edge detection to find texture
            edges = np.array(gray.filter(ImageFilter.FIND_EDGES))
            
            # Calculate edge density
            edge_density = np.sum(edges > 20) / edges.size  # Lower threshold
            
            # Low edge density indicates smoothness
            smoothness = 1 - min(edge_density * 10, 1)  # More sensitive
            
            # Also check variance
            local_vars = []
            step = 8  # Sample more frequently
            
            for i in range(2, img_array.shape[0] - 2, step):
                for j in range(2, img_array.shape[1] - 2, step):
                    window = img_array[i-2:i+3, j-2:j+3]
                    local_vars.append(np.var(window))
            
            if local_vars:
                avg_var = np.mean(local_vars)
                # Low variance indicates smoothness
                var_smoothness = 1 - min(avg_var / 300, 1)  # More sensitive threshold
                
                # Combine both metrics
                return (smoothness * 0.5 + var_smoothness * 0.5)
            
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
            
            # Check for repeating patterns
            pattern_score = 0
            correlations = []
            
            # Check horizontal patterns
            for row in range(0, small.shape[0], 6):
                row_data = small[row, :]
                if len(row_data) > 20:
                    # Simple autocorrelation
                    for shift in range(4, len(row_data)//3, 4):
                        if shift < len(row_data):
                            try:
                                correlation = np.corrcoef(
                                    row_data[:-shift], 
                                    row_data[shift:]
                                )[0, 1]
                                correlations.append(correlation)
                                if correlation > 0.8:
                                    pattern_score += 0.2
                                elif correlation > 0.6:
                                    pattern_score += 0.1
                            except:
                                pass
            
            # Check vertical patterns
            for col in range(0, small.shape[1], 6):
                col_data = small[:, col]
                if len(col_data) > 20:
                    for shift in range(4, len(col_data)//3, 4):
                        if shift < len(col_data):
                            try:
                                correlation = np.corrcoef(
                                    col_data[:-shift], 
                                    col_data[shift:]
                                )[0, 1]
                                correlations.append(correlation)
                                if correlation > 0.8:
                                    pattern_score += 0.2
                                elif correlation > 0.6:
                                    pattern_score += 0.1
                            except:
                                pass
            
            # Also check for high average correlation
            if correlations:
                avg_correlation = np.mean([abs(c) for c in correlations if not np.isnan(c)])
                if avg_correlation > 0.5:
                    pattern_score += 0.3
                elif avg_correlation > 0.3:
                    pattern_score += 0.15
            
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
            height, width = edge_array.shape
            
            # Sample edge pixels
            edge_pixels = np.where(edge_array > 60)  # Lower threshold
            
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
                
                if alignment_ratio > 0.3:
                    artifacts.append("Unnaturally straight edges detected")
                    score += 25
                elif alignment_ratio > 0.2:
                    artifacts.append("Some artificial edge patterns")
                    score += 15
                elif alignment_ratio > 0.15:
                    artifacts.append("Minor edge alignment patterns")
                    score += 8
            
            # Check for edge consistency issues
            edge_variance = np.var(edge_array[edge_array > 40])
            if edge_variance < 100:
                artifacts.append("Suspiciously consistent edge strength")
                score += 20
            elif edge_variance < 200:
                artifacts.append("Uniform edge characteristics")
                score += 10
            
            # Check for too-perfect curves
            if self._check_perfect_curves(edge_array):
                artifacts.append("Mathematically perfect curves detected")
                score += 15
            
            return artifacts, score
            
        except Exception as e:
            logger.error(f"Edge artifact check error: {e}")
            return [], 0
    
    def _check_gradient_perfection(self, image: Any) -> float:
        """Check for perfect gradients"""
        if not self.pil_available or not np:
            return 0.4
            
        try:
            # Convert to grayscale
            gray = np.array(image.convert('L'))
            
            # Check multiple directions for gradients
            perfection_scores = []
            
            # Horizontal gradient check
            for row in range(0, gray.shape[0], max(1, gray.shape[0]//10)):
                row_data = gray[row, :]
                if len(row_data) > 10:
                    # Calculate differences
                    diffs = np.diff(row_data.astype(float))
                    if len(diffs) > 0:
                        # Check consistency of differences
                        if np.std(diffs) > 0:
                            consistency = 1 - min(np.std(diffs) / (np.mean(np.abs(diffs)) + 1), 1)
                        else:
                            consistency = 1.0  # Perfect gradient
                        perfection_scores.append(consistency)
            
            # Vertical gradient check
            for col in range(0, gray.shape[1], max(1, gray.shape[1]//10)):
                col_data = gray[:, col]
                if len(col_data) > 10:
                    diffs = np.diff(col_data.astype(float))
                    if len(diffs) > 0:
                        if np.std(diffs) > 0:
                            consistency = 1 - min(np.std(diffs) / (np.mean(np.abs(diffs)) + 1), 1)
                        else:
                            consistency = 1.0
                        perfection_scores.append(consistency)
            
            if perfection_scores:
                # Return high percentile to catch perfect gradients
                sorted_scores = sorted(perfection_scores)
                index = int(len(sorted_scores) * 0.8)
                return min(sorted_scores[index], 1.0)
            
            return 0.4
            
        except Exception as e:
            logger.error(f"Gradient check error: {e}")
            return 0.4
    
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
            h_projection = np.sum(edge_array > 60, axis=1)
            v_projection = np.sum(edge_array > 60, axis=0)
            
            # Find peaks (potential grid lines)
            h_peaks = self._find_regular_peaks(h_projection)
            v_peaks = self._find_regular_peaks(v_projection)
            
            # Calculate grid strength
            grid_strength = 0
            
            # If we find regular peaks in both directions, it's likely a grid
            if len(h_peaks) > 2 and len(v_peaks) > 2:
                # Calculate regularity score
                h_regularity = self._calculate_spacing_regularity(h_peaks)
                v_regularity = self._calculate_spacing_regularity(v_peaks)
                
                grid_strength = (h_regularity + v_regularity) / 2
                
                if grid_strength > 0.5:
                    return True, grid_strength
            
            # Also check for subtle grids
            elif len(h_peaks) > 1 or len(v_peaks) > 1:
                if len(h_peaks) > 1:
                    h_regularity = self._calculate_spacing_regularity(h_peaks)
                else:
                    h_regularity = 0
                    
                if len(v_peaks) > 1:
                    v_regularity = self._calculate_spacing_regularity(v_peaks)
                else:
                    v_regularity = 0
                    
                grid_strength = max(h_regularity, v_regularity) * 0.7
            
            return False, grid_strength
            
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
            strong_edges = edge_array > 100
            
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
                        
                        # Measure edge width
                        width = 1
                        for d in range(1, 10):
                            if (x + d < edge_array.shape[1] and 
                                edge_array[y, x + d] > 80):
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
                        elif width_variance < 1.0:
                            anomalies.append("Uniform edge characteristics")
                        
                        if avg_width > 3:
                            anomalies.append("Unusually thick edges detected")
            
            # Check for impossible edge transitions
            edge_continuity = self._check_edge_continuity(edge_array)
            if edge_continuity < 0.4:
                anomalies.append("Discontinuous edges detected")
            elif edge_continuity < 0.6:
                anomalies.append("Some edge discontinuities")
            
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
            block_size = 24  # Smaller blocks for more sensitivity
            uniformity_scores = []
            
            for y in range(0, gray.shape[0] - block_size, block_size):
                for x in range(0, gray.shape[1] - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    
                    # Calculate local statistics
                    block_std = np.std(block)
                    block_range = np.max(block) - np.min(block)
                    
                    # Low variance and range indicate uniform texture
                    if block_std < 3 and block_range < 10:
                        uniformity_scores.append(1)
                    elif block_std < 5 and block_range < 20:
                        uniformity_scores.append(0.5)
                    else:
                        uniformity_scores.append(0)
            
            if uniformity_scores:
                uniformity_ratio = sum(uniformity_scores) / len(uniformity_scores)
                
                if uniformity_ratio > 0.3:
                    anomalies.append(f"Large uniform texture areas ({uniformity_ratio*100:.0f}% of image)")
                elif uniformity_ratio > 0.2:
                    anomalies.append(f"Some uniform texture areas detected")
                
                # Check for repeating texture patterns
                if self._has_repeating_textures(gray):
                    anomalies.append("Repeating texture patterns detected")
            
            # Check for too-perfect textures
            texture_perfection = self._check_texture_perfection(gray)
            if texture_perfection > 0.7:
                anomalies.append("Unnaturally perfect texture rendering")
            elif texture_perfection > 0.5:
                anomalies.append("Suspiciously clean textures")
            
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
                    score += 40
                    
            elif model == 'dalle':
                # DALL-E can have grid artifacts and specific color handling
                grid_detected, grid_strength = self._has_grid_pattern(image)
                if grid_detected:
                    score += 35
                elif grid_strength > 0.3:
                    score += 20
                    
            elif model == 'stable_diffusion':
                # SD often has specific noise patterns
                if self._check_sd_noise_pattern(image):
                    score += 35
            
            # Generic pattern checks
            if 'perfect symmetry' in patterns:
                symmetry = self._check_symmetry(image)
                if symmetry > 0.8:
                    score += 25
                elif symmetry > 0.6:
                    score += 15
                elif symmetry > 0.4:
                    score += 8
                    
            if 'unrealistic lighting' in patterns:
                if self._check_lighting_consistency(image):
                    score += 20
                    
            if 'texture artifacts' in patterns:
                if self._has_texture_anomalies(image):
                    score += 20
            
            return min(score, 70)  # Cap to require other evidence
            
        except Exception as e:
            logger.error(f"Model pattern check error: {e}")
            return 0
    
    def _basic_analysis(self, image_data: str) -> Dict[str, Any]:
        """Enhanced basic analysis when PIL is not available"""
        # More intelligent heuristics based on base64 data
        data_size = len(image_data)
        
        # Start with higher base probability for modern context
        ai_probability = 60
        
        # Check data size patterns
        if data_size > 2000000:  # > 2MB base64 (high resolution, common in AI)
            ai_probability += 15
        elif data_size > 1000000:  # > 1MB
            ai_probability += 10
        elif data_size < 100000:  # Very small (unlikely for modern AI)
            ai_probability -= 10
        
        # Check for patterns in base64 encoding
        sample = image_data[:5000]
        
        # Look for repetitive patterns (common in AI images)
        pattern_count = 0
        for i in range(0, len(sample) - 100, 20):
            pattern = sample[i:i+10]
            if sample.count(pattern) > 3:
                pattern_count += 1
        
        if pattern_count > 10:
            ai_probability += 15
        elif pattern_count > 5:
            ai_probability += 8
        
        # Check for specific base64 patterns common in AI images
        # AI images often have certain byte patterns when encoded
        if 'AAAA' in sample or 'ffff' in sample or '////' in sample:
            ai_probability += 5
        
        return {
            'ai_probability': min(ai_probability, 85),
            'note': 'Limited analysis. For accurate results, ensure Pillow (PIL) is installed.',
            'data_size': data_size,
            'basic_indicators': pattern_count
        }
    
    def _check_perfection_indicators(self, image: Any) -> float:
        """Check for various perfection indicators common in AI images"""
        if not self.pil_available or not np:
            return 0.5
        
        try:
            perfection_score = 0
            indicators = 0
            
            # Check for perfect circles/shapes
            if self._has_perfect_shapes(image):
                perfection_score += 0.25
                indicators += 1
            
            # Check for too-clean backgrounds
            if self._has_perfect_background(image):
                perfection_score += 0.2
                indicators += 1
            
            # Check for unrealistic detail consistency
            if self._has_consistent_detail_level(image):
                perfection_score += 0.2
                indicators += 1
            
            # Check for perfect lighting
            if self._check_lighting_consistency(image):
                perfection_score += 0.2
                indicators += 1
            
            # Check for lack of natural imperfections
            if self._lacks_natural_imperfections(image):
                perfection_score += 0.15
                indicators += 1
            
            # Boost score if multiple indicators
            if indicators >= 3:
                perfection_score *= 1.2
            
            return min(perfection_score, 1.0)
            
        except Exception as e:
            logger.error(f"Perfection check error: {e}")
            return 0.5
    
    def _check_ai_color_patterns(self, image: Any) -> float:
        """Check for color patterns typical of AI-generated images"""
        if not self.pil_available or not np:
            return 0.3
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            score = 0
            
            # Check for oversaturated colors (common in AI)
            saturation = self._calculate_saturation(img_array)
            if saturation > 0.7:
                score += 0.3
            elif saturation > 0.6:
                score += 0.2
            
            # Check for unnatural color transitions
            color_transitions = self._analyze_color_transitions(img_array)
            if color_transitions < 0.3:  # Too smooth
                score += 0.3
            elif color_transitions < 0.5:
                score += 0.15
            
            # Check for specific color biases common in AI
            color_bias = self._check_ai_color_bias(img_array)
            score += color_bias * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"AI color pattern check error: {e}")
            return 0.3
    
    def _check_ai_rendering_artifacts(self, image: Any) -> List[str]:
        """Check for AI-specific rendering artifacts"""
        artifacts = []
        
        try:
            # Check for telltale signs of AI rendering
            if self._has_latent_space_artifacts(image):
                artifacts.append("Latent space artifacts detected")
            
            if self._has_gan_artifacts(image):
                artifacts.append("GAN-style artifacts present")
            
            if self._has_diffusion_artifacts(image):
                artifacts.append("Diffusion model artifacts detected")
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Rendering artifact check error: {e}")
            return artifacts
    
    # Additional helper methods for enhanced detection
    
    def _check_perfect_curves(self, edge_array: np.ndarray) -> bool:
        """Check for mathematically perfect curves"""
        # Simplified check - looks for too-smooth curves
        try:
            # Sample some edge points and check curvature consistency
            edge_points = np.where(edge_array > 100)
            if len(edge_points[0]) > 30:
                # This is a simplified version
                return False  # Would need more complex implementation
            return False
        except:
            return False
    
    def _check_texture_perfection(self, gray_array: np.ndarray) -> float:
        """Check how perfect/artificial the texture is"""
        try:
            # Calculate local entropy
            perfection_score = 0
            
            # Check for too-regular patterns
            fft = np.fft.fft2(gray_array)
            fft_mag = np.abs(fft)
            
            # Regular patterns show up as peaks in FFT
            peaks = np.sum(fft_mag > np.mean(fft_mag) * 5)
            if peaks > 20:
                perfection_score += 0.4
            elif peaks > 10:
                perfection_score += 0.2
            
            return perfection_score
        except:
            return 0
    
    def _has_perfect_shapes(self, image: Any) -> bool:
        """Check for perfect geometric shapes"""
        # This would use shape detection algorithms
        # Simplified version
        return False
    
    def _has_perfect_background(self, image: Any) -> bool:
        """Check for unnaturally clean backgrounds"""
        try:
            # Check edges of image for uniform color
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Sample border regions
            border_size = min(20, h//10, w//10)
            
            top_border = img_array[:border_size, :]
            bottom_border = img_array[-border_size:, :]
            left_border = img_array[:, :border_size]
            right_border = img_array[:, -border_size:]
            
            # Check variance
            variances = [
                np.var(top_border),
                np.var(bottom_border),
                np.var(left_border),
                np.var(right_border)
            ]
            
            # Low variance in borders suggests perfect background
            avg_variance = np.mean(variances)
            return avg_variance < 100
            
        except:
            return False
    
    def _has_consistent_detail_level(self, image: Any) -> bool:
        """Check if detail level is unnaturally consistent"""
        try:
            # Divide image into regions and check detail consistency
            gray = np.array(image.convert('L'))
            h, w = gray.shape
            
            region_size = min(h, w) // 4
            detail_levels = []
            
            for i in range(0, h - region_size, region_size):
                for j in range(0, w - region_size, region_size):
                    region = gray[i:i+region_size, j:j+region_size]
                    # Use edge detection as proxy for detail
                    edges = np.array(Image.fromarray(region).filter(ImageFilter.FIND_EDGES))
                    detail_levels.append(np.mean(edges))
            
            if detail_levels:
                # Check if detail levels are too consistent
                detail_variance = np.var(detail_levels)
                mean_detail = np.mean(detail_levels)
                if mean_detail > 0:
                    cv = detail_variance / mean_detail
                    return cv < 0.3  # Low coefficient of variation
            
            return False
        except:
            return False
    
    def _lacks_natural_imperfections(self, image: Any) -> bool:
        """Check for lack of natural imperfections"""
        try:
            # Look for things like dust, scratches, noise that appear in real photos
            gray = np.array(image.convert('L'))
            
            # Check for very small variations (natural noise)
            small_variations = np.sum(np.abs(np.diff(gray, axis=0)) < 3) + \
                              np.sum(np.abs(np.diff(gray, axis=1)) < 3)
            total_pixels = gray.size
            
            # Too few small variations suggests lack of natural noise
            variation_ratio = small_variations / total_pixels
            return variation_ratio > 0.95  # Too clean
            
        except:
            return False
    
    def _analyze_color_transitions(self, img_array: np.ndarray) -> float:
        """Analyze how natural color transitions are"""
        try:
            # Check color gradients
            transitions = []
            
            # Sample some rows and columns
            for i in range(0, img_array.shape[0], img_array.shape[0]//10):
                row = img_array[i, :, :]
                diffs = np.diff(row, axis=0)
                transitions.append(np.mean(np.abs(diffs)))
            
            for j in range(0, img_array.shape[1], img_array.shape[1]//10):
                col = img_array[:, j, :]
                diffs = np.diff(col, axis=0)
                transitions.append(np.mean(np.abs(diffs)))
            
            if transitions:
                # Return normalized transition score
                avg_transition = np.mean(transitions)
                return min(avg_transition / 50, 1.0)
            
            return 0.5
        except:
            return 0.5
    
    def _check_ai_color_bias(self, img_array: np.ndarray) -> float:
        """Check for color biases common in AI images"""
        try:
            # AI images often have certain color biases
            r_mean = np.mean(img_array[:,:,0])
            g_mean = np.mean(img_array[:,:,1])
            b_mean = np.mean(img_array[:,:,2])
            
            # Check for common AI color biases
            bias_score = 0
            
            # Teal/cyan bias (common in some AI models)
            if g_mean > r_mean * 1.1 and b_mean > r_mean * 1.1:
                bias_score += 0.3
            
            # Purple/magenta bias
            if r_mean > g_mean * 1.1 and b_mean > g_mean * 1.1:
                bias_score += 0.3
            
            # Check for oversaturated specific channels
            if any(m > 180 for m in [r_mean, g_mean, b_mean]):
                bias_score += 0.2
            
            return min(bias_score, 0.8)
        except:
            return 0
    
    def _has_latent_space_artifacts(self, image: Any) -> bool:
        """Check for latent space artifacts"""
        # This would be more complex in reality
        # Looking for specific patterns that emerge from latent space
        return False
    
    def _has_gan_artifacts(self, image: Any) -> bool:
        """Check for GAN-specific artifacts"""
        # Would look for mode collapse patterns, checkerboard artifacts, etc.
        return False
    
    def _has_diffusion_artifacts(self, image: Any) -> bool:
        """Check for diffusion model artifacts"""
        # Would look for specific noise patterns from diffusion process
        return False
    
    # Keep all the existing helper methods...
    # [All other methods remain the same as in the original file]
    
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
                for _ in range(15):  # More samples
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
                        
                        # Count sudden changes that might indicate banding
                        sudden_changes = np.sum(np.abs(diffs) > 5)
                        smooth_areas = np.sum(np.abs(diffs) < 2)
                        
                        # Banding shows as sudden changes amid smooth areas
                        if smooth_areas > len(diffs) * 0.6 and sudden_changes > 3:
                            banding_scores.append(1)
                        elif smooth_areas > len(diffs) * 0.4 and sudden_changes > 2:
                            banding_scores.append(0.5)
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
                if mean_noise > 3 and noise_variance < mean_noise * 0.5:
                    return 0.7
                elif mean_noise > 2 and noise_variance < mean_noise * 0.7:
                    return 0.5
                elif mean_noise < 1:
                    # Too little noise is also suspicious
                    return 0.4
            
            return 0.2
            
        except Exception as e:
            logger.error(f"Noise pattern check error: {e}")
            return 0
    
    def _find_regular_peaks(self, projection: np.ndarray, min_distance: int = 8) -> List[int]:
        """Find regularly spaced peaks in projection"""
        peaks = []
        threshold = np.mean(projection) + np.std(projection) * 0.5  # Lower threshold
        
        i = 0
        while i < len(projection):
            if projection[i] > threshold:
                # Find local maximum
                local_max = i
                while i < len(projection) and projection[i] > threshold * 0.7:
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
        if edge_density > 0.08:
            return 0.8
        elif edge_density > 0.04:
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
                    
                    if diff < 8:  # Very similar
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
                if np.std(freq_diffs) < np.mean(np.abs(freq_diffs)) * 0.25:
                    score += 25
                elif np.std(freq_diffs) < np.mean(np.abs(freq_diffs)) * 0.4:
                    score += 15
            
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
                if zero_bins > 160:
                    anomaly_score += 0.25
                elif zero_bins > 120:
                    anomaly_score += 0.15
                
                # 2. Unnatural spikes
                mean_count = sum(channel_hist) / 256
                spike_count = sum(1 for h in channel_hist if h > mean_count * 8)
                if spike_count > 3:
                    anomaly_score += 0.2
                
                # 3. Too smooth histogram (over-processed)
                hist_diffs = [abs(channel_hist[i] - channel_hist[i+1]) 
                             for i in range(255)]
                if sum(hist_diffs) < mean_count * 40:
                    anomaly_score += 0.15
            
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
            edge_strength = np.mean(edges[edges > 40])
            
            # Check color characteristics
            if image.mode == 'RGB':
                img_array = np.array(image)
                
                # Midjourney often has rich, saturated colors
                saturation = self._calculate_saturation(img_array)
                
                if edge_strength > 80 and saturation > 0.55:
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
            if 2.5 < noise_std < 10 and noise_mean < 1.5:
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
            
            if region_size < 10:
                return False
            
            region_means = []
            region_stds = []
            
            for i in range(0, h - region_size, region_size):
                for j in range(0, w - region_size, region_size):
                    region = luminance[i:i+region_size, j:j+region_size]
                    region_means.append(np.mean(region))
                    region_stds.append(np.std(region))
            
            if region_means:
                # Check if lighting is too consistent across regions
                mean_variance = np.var(region_means)
                
                # Unnaturally consistent if variance is very low
                if mean_variance < 50:
                    return True
                
                # Also check if standard deviations are too similar
                std_variance = np.var(region_stds)
                if std_variance < 10:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Lighting consistency check error: {e}")
            return False
    
    def _calculate_saturation(self, img_array: np.ndarray) -> float:
        """Calculate average saturation of image"""
        try:
            # Convert RGB to HSV approximation
            r = img_array[:,:,0] / 255.0
            g = img_array[:,:,1] / 255.0
            b = img_array[:,:,2] / 255.0
            
            # Calculate max and min for each pixel
            cmax = np.maximum(r, np.maximum(g, b))
            cmin = np.minimum(r, np.minimum(g, b))
            
            # Calculate saturation
            diff = cmax - cmin
            
            # Avoid division by zero
            saturation = np.where(cmax > 0, diff / cmax, 0)
            
            return np.mean(saturation)
            
        except Exception as e:
            logger.error(f"Saturation calculation error: {e}")
            return 0.5
    
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
            if std_diff < 3:
                anomalies.append("Extremely uniform compression artifacts")
            elif std_diff < 6:
                anomalies.append("Uniform compression artifacts detected")
            
            # Check for areas with no compression artifacts (too perfect)
            flat_areas = np.sum(diff_array < 2) / diff_array.size
            if flat_areas > 0.4:
                anomalies.append("Large areas with no compression artifacts")
            elif flat_areas > 0.25:
                anomalies.append("Suspiciously perfect areas detected")
            
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
                    if avg_block_diff < 2:
                        artifacts.append("Extremely low JPEG block artifacts")
                    elif avg_block_diff < 4:
                        artifacts.append("Unusually low JPEG block artifacts")
                    elif avg_block_diff > 60:
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
                if zero_bins > 120:
                    anomalies.append(f"{channel} channel has unnatural gaps")
                elif zero_bins > 80:
                    anomalies.append(f"{channel} channel has some gaps")
                
                # Check for unnatural peaks
                if max_bin > avg_bin * 60:
                    anomalies.append(f"{channel} channel has extreme color concentration")
                elif max_bin > avg_bin * 40:
                    anomalies.append(f"{channel} channel has unnatural color concentration")
            
            # Check overall color count
            unique_colors = len(set(image.getdata()))
            total_pixels = image.size[0] * image.size[1]
            color_ratio = unique_colors / total_pixels
            
            if color_ratio < 0.05:
                anomalies.append("Extremely low color diversity")
            elif color_ratio < 0.1:
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
                if avg_diff > 60:
                    inconsistencies.append("Strong scale-dependent edge artifacts")
                elif avg_diff > 45:
                    inconsistencies.append("Scale-dependent edge artifacts")
            
            return {'inconsistencies': inconsistencies}
            
        except Exception as e:
            logger.error(f"Edge coherence error: {e}")
            return {'inconsistencies': []}
    
    def _calculate_forensic_probability(self, ela: Dict, compression: Dict, 
                                      color: Dict, edge: Dict) -> float:
        """Calculate probability based on forensic analysis"""
        score = 40  # Higher base score for forensic analysis
        
        # Add points for each anomaly type
        score += len(ela.get('anomalies', [])) * 18
        score += len(compression.get('ai_artifacts', [])) * 15
        score += len(color.get('anomalies', [])) * 10
        score += len(edge.get('inconsistencies', [])) * 12
        
        # Additional scoring based on specific findings
        if ela.get('mean_error_level', 10) < 2:
            score += 20  # Very low error levels are highly suspicious
        elif ela.get('mean_error_level', 10) < 4:
            score += 10
        
        if ela.get('error_level_variance', 10) < 2:
            score += 15  # Too uniform
        elif ela.get('error_level_variance', 10) < 4:
            score += 8
        
        return min(score, 98)
    
    def _create_image_summary(self, ai_probability: float, artifacts: Dict) -> str:
        """Create summary for image analysis"""
        artifact_count = artifacts.get('artifact_count', 0)
        
        if ai_probability >= 80:
            return f"This image shows strong signs of AI generation with {artifact_count} artifacts detected. High probability of being created by AI image generation tools like Midjourney, DALL-E, or Stable Diffusion."
        elif ai_probability >= 60:
            return f"This image likely contains AI-generated elements. {artifact_count} suspicious patterns were found that are common in AI-generated images. The image may be fully AI-generated or heavily AI-processed."
        elif ai_probability >= 40:
            return f"Mixed indicators present with {artifact_count} potential artifacts. The image shows some characteristics of AI generation but may also have human-created elements or be a heavily edited photograph."
        else:
            return f"This image appears to be predominantly authentic with minimal signs of AI generation. Only {artifact_count} minor anomalies detected. Likely a genuine photograph or human-created digital art."
    
    def _detect_model(self, metadata: Dict, artifacts: Dict, pixels: Dict) -> str:
        """Detect which AI model might have generated the image"""
        # Check for software in metadata first
        if metadata.get('detected_software'):
            software = ' '.join(metadata['detected_software'])
            
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
            elif pixels and pixels.get('gradient_perfection', 0) > 0.75:
                return 'Midjourney (suspected)'
            elif 'noise patterns' in artifacts_str.lower():
                return 'Stable Diffusion (suspected)'
        
        # Check for high quality characteristics
        if pixels and pixels.get('color_uniformity', 0) < 0.3 and pixels.get('gradient_perfection', 0) > 0.7:
            return 'Midjourney (suspected)'
        
        if metadata.get('ai_indicators', 0) > 0:
            return 'Unknown AI Model'
        
        return 'Not Detected'
    
    def _summarize_artifacts(self, ela: Dict, compression: Dict, 
                           color: Dict, edge: Dict) -> List[str]:
        """Summarize all detected artifacts"""
        artifacts = []
        
        artifacts.extend(ela.get('anomalies', []))
        artifacts.extend(compression.get('ai_artifacts', []))
        artifacts.extend(color.get('anomalies', []))
        artifacts.extend(edge.get('inconsistencies', []))
        
        return artifacts[:5]  # Return top 5
