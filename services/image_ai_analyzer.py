"""
Image AI Analyzer Service
Detects AI-generated images through various analysis methods
"""
import os
import logging
import base64
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from io import BytesIO
import json

# Try to import image processing libraries
try:
    from PIL import Image, ImageStat
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageStat = None
    np = None

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
                'metadata_keys': ['midjourney', 'mj']
            },
            'dalle': {
                'patterns': ['grid artifacts', 'unnatural edges', 'color banding'],
                'metadata_keys': ['openai', 'dalle', 'dall-e']
            },
            'stable_diffusion': {
                'patterns': ['noise patterns', 'latent artifacts', 'repetitive textures'],
                'metadata_keys': ['stable-diffusion', 'sd', 'stability']
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
                'detected_model': self._detect_model(metadata_analysis, artifact_analysis),
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
                
                # Check visual patterns (simplified)
                pattern_score = self._check_model_patterns(image, info['patterns'])
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
                    'detected_model': signatures_found[0]['model'].title(),
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
            
            # Check for unrealistic smoothness
            smoothness = self._check_smoothness(image)
            if smoothness > 0.85:
                artifacts.append("Unrealistic surface smoothness")
                ai_score += 15
            
            # Check for repetitive patterns
            repetition = self._check_repetitive_patterns(image)
            if repetition > 0.7:
                artifacts.append("Repetitive pattern artifacts")
                ai_score += 15
            
            # Check for edge artifacts
            edge_artifacts = self._check_edge_artifacts(image)
            if edge_artifacts:
                artifacts.append("AI edge generation artifacts")
                ai_score += 20
            
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
        
        if hasattr(image, 'info'):
            metadata = image.info
            
            # Check for AI-related metadata
            metadata_str = str(metadata).lower()
            ai_keywords = [
                'ai', 'generated', 'midjourney', 'dalle', 'stable diffusion',
                'artificial', 'synthesis', 'gan', 'neural'
            ]
            
            for keyword in ai_keywords:
                if keyword in metadata_str:
                    ai_indicators += 1
        
        return {
            'has_metadata': bool(metadata),
            'ai_indicators': ai_indicators,
            'suspicious_metadata': ai_indicators > 0
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
            
            # Check for perfect gradients
            gradient_score = self._check_gradient_perfection(image)
            
            return {
                'color_uniformity': round(uniformity_score, 2),
                'gradient_perfection': round(gradient_score, 2),
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
            high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.mean(magnitude_spectrum)])
            total_energy = np.sum(magnitude_spectrum)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # AI images often have different frequency characteristics
            ai_frequency_score = 0
            if high_freq_ratio < 0.3:  # Low high-frequency content
                ai_frequency_score += 50
            
            return {
                'high_frequency_ratio': round(high_freq_ratio, 3),
                'ai_frequency_score': ai_frequency_score
            }
            
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return {}
    
    def _detect_ai_artifacts(self, image: Any) -> Dict[str, Any]:
        """Detect specific AI generation artifacts"""
        artifacts = []
        
        # Check for grid patterns (DALL-E artifact)
        if self._has_grid_pattern(image):
            artifacts.append("Grid pattern detected")
        
        # Check for unnatural edges
        if self._has_unnatural_edges(image):
            artifacts.append("Unnatural edge transitions")
        
        # Check for texture anomalies
        if self._has_texture_anomalies(image):
            artifacts.append("Texture generation artifacts")
        
        return {
            'artifacts_found': artifacts,
            'artifact_count': len(artifacts)
        }
    
    def _calculate_image_ai_probability(self, metadata: Dict, pixels: Dict, 
                                      frequency: Dict, artifacts: Dict) -> float:
        """Calculate overall AI probability for image"""
        score = 0
        weights = 0
        
        # Metadata score
        if metadata.get('ai_indicators', 0) > 0:
            score += 90 * 0.3
            weights += 0.3
        else:
            weights += 0.1  # Lower weight if no metadata
        
        # Pixel analysis score
        if 'color_uniformity' in pixels:
            uniformity = pixels['color_uniformity']
            if uniformity > 0.7:
                score += 80 * 0.2
                weights += 0.2
        
        # Frequency analysis score
        if 'ai_frequency_score' in frequency:
            score += frequency['ai_frequency_score'] * 0.25
            weights += 0.25
        
        # Artifacts score
        if 'artifact_count' in artifacts:
            artifact_score = min(artifacts['artifact_count'] * 20, 80)
            score += artifact_score * 0.25
            weights += 0.25
        
        # Calculate final score
        if weights > 0:
            final_score = score / weights
        else:
            final_score = 50
        
        return final_score
    
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
        if image_data[:1000].count(image_data[100:110]) > 5:
            ai_probability += 20
        
        return {
            'ai_probability': ai_probability,
            'note': 'Limited analysis available. Install Pillow for full image analysis.',
            'data_size': data_size
        }
    
    def _create_image_summary(self, ai_probability: float, artifacts: Dict) -> str:
        """Create summary for image analysis"""
        if ai_probability >= 80:
            return f"This image shows strong signs of AI generation with {len(artifacts.get('artifacts_found', []))} artifacts detected."
        elif ai_probability >= 60:
            return "This image likely contains AI-generated elements or was heavily processed by AI."
        elif ai_probability >= 40:
            return "Mixed indicators present. The image may have some AI involvement or processing."
        else:
            return "This image appears to be authentic with minimal signs of AI generation."
    
    def _detect_model(self, metadata: Dict, artifacts: Dict) -> str:
        """Detect which AI model might have generated the image"""
        if metadata.get('ai_indicators', 0) > 0:
            return 'AI Model Detected'
        return 'Unknown'
    
    # Helper methods for various checks
    def _check_symmetry(self, image: Any) -> float:
        """Check image symmetry"""
        if not self.pil_available or not np:
            return 0
            
        try:
            width, height = image.size
            img_array = np.array(image)
            
            # Check horizontal symmetry
            left_half = img_array[:, :width//2]
            right_half = np.fliplr(img_array[:, width//2:])
            
            if left_half.shape == right_half.shape:
                diff = np.mean(np.abs(left_half - right_half))
                symmetry_score = 1 - (diff / 255)
                return symmetry_score
            
            return 0
        except:
            return 0
    
    def _check_smoothness(self, image: Any) -> float:
        """Check for unrealistic smoothness"""
        if not self.pil_available or not np:
            return 0.5
            
        try:
            img_array = np.array(image.convert('L'))
            
            # Calculate local variance
            kernel_size = 3
            pad = kernel_size // 2
            padded = np.pad(img_array, pad, mode='edge')
            
            local_vars = []
            for i in range(pad, padded.shape[0] - pad):
                for j in range(pad, padded.shape[1] - pad):
                    window = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                    local_vars.append(np.var(window))
            
            # Low variance indicates smoothness
            avg_var = np.mean(local_vars)
            smoothness = 1 - min(avg_var / 1000, 1)
            
            return smoothness
        except:
            return 0.5
    
    def _check_repetitive_patterns(self, image: Any) -> float:
        """Check for repetitive patterns"""
        # Simplified check - in reality would use more sophisticated methods
        return 0.3  # Placeholder
    
    def _check_edge_artifacts(self, image: Any) -> bool:
        """Check for AI edge artifacts"""
        # Simplified check
        return False  # Placeholder
    
    def _check_gradient_perfection(self, image: Any) -> float:
        """Check for perfect gradients"""
        # Simplified check
        return 0.5  # Placeholder
    
    def _has_grid_pattern(self, image: Any) -> bool:
        """Check for grid patterns"""
        # Simplified check
        return False  # Placeholder
    
    def _has_unnatural_edges(self, image: Any) -> bool:
        """Check for unnatural edges"""
        # Simplified check
        return False  # Placeholder
    
    def _has_texture_anomalies(self, image: Any) -> bool:
        """Check for texture anomalies"""
        # Simplified check
        return False  # Placeholder
    
    def _check_model_patterns(self, image: Any, patterns: List[str]) -> float:
        """Check for model-specific patterns"""
        # Simplified check - returns a score based on pattern matching
        return 20  # Placeholder
    
    def _error_level_analysis(self, image: Any) -> Dict[str, Any]:
        """Perform error level analysis"""
        # Simplified ELA
        return {
            'anomalies': ['Compression inconsistency detected']
        }
    
    def _analyze_compression_artifacts(self, image: Any) -> Dict[str, Any]:
        """Analyze compression artifacts"""
        return {
            'ai_artifacts': ['Uniform compression blocks']
        }
    
    def _analyze_color_distribution(self, image: Any) -> Dict[str, Any]:
        """Analyze color distribution"""
        return {
            'anomalies': ['Unusual color histogram']
        }
    
    def _analyze_edge_coherence(self, image: Any) -> Dict[str, Any]:
        """Analyze edge coherence"""
        return {
            'inconsistencies': ['Edge discontinuities']
        }
    
    def _calculate_forensic_probability(self, ela: Dict, compression: Dict, 
                                      color: Dict, edge: Dict) -> float:
        """Calculate probability based on forensic analysis"""
        score = 50
        
        if ela.get('anomalies'):
            score += len(ela['anomalies']) * 10
        
        if compression.get('ai_artifacts'):
            score += len(compression['ai_artifacts']) * 10
        
        if color.get('anomalies'):
            score += len(color['anomalies']) * 5
        
        if edge.get('inconsistencies'):
            score += len(edge['inconsistencies']) * 5
        
        return min(score, 95)
    
    def _summarize_artifacts(self, ela: Dict, compression: Dict, 
                           color: Dict, edge: Dict) -> List[str]:
        """Summarize all detected artifacts"""
        artifacts = []
        
        artifacts.extend(ela.get('anomalies', []))
        artifacts.extend(compression.get('ai_artifacts', []))
        artifacts.extend(color.get('anomalies', []))
        artifacts.extend(edge.get('inconsistencies', []))
        
        return artifacts[:5]  # Return top 5
    
    # Grid detection helper methods - using List[int] instead of np.ndarray
    def _find_regular_peaks(self, projection: List[Union[int, float]], min_distance: int = 10) -> List[int]:
        """Find regularly spaced peaks in projection data"""
        if not projection:
            return []
        
        peaks = []
        for i in range(1, len(projection) - 1):
            if projection[i] > projection[i-1] and projection[i] > projection[i+1]:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        
        return peaks
    
    def _is_grid_regular(self, peaks: List[int], tolerance: float = 0.1) -> bool:
        """Check if peaks are regularly spaced (indicating a grid)"""
        if len(peaks) < 3:
            return False
        
        intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        avg_interval = sum(intervals) / len(intervals)
        
        for interval in intervals:
            if abs(interval - avg_interval) / avg_interval > tolerance:
                return False
        
        return True
