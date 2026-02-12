import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ImagePreprocessor:
    """
    Simple image preprocessing for face detection.
    Assesses quality and applies enhancements when needed.
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.quality_thresholds = {
            'brightness': {'low': 0.3, 'high': 0.7},
            'contrast': {'low': 0.3, 'high': 0.6},
            'sharpness': {'low': 0.2, 'high': 0.5}
        }

    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality metrics.
        
        Returns dict with:
            - brightness: 0-1 (0=dark, 1=bright)
            - contrast: 0-1 (0=low, 1=high)
            - sharpness: 0-1 (0=blurry, 1=sharp)
            - resolution_score: 0-1 (based on face region size)
        """
        if image is None or image.size == 0:
            return {
                'brightness': 0.0,
                'contrast': 0.0,
                'sharpness': 0.0,
                'resolution_score': 0.0,
                'overall': 0.0
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray) / 255.0

        contrast = np.std(gray) / 128.0
        contrast = min(1.0, contrast)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500.0)

        h, w = image.shape[:2]
        face_region_size = min(h, w) / 300.0
        resolution_score = min(1.0, face_region_size)

        overall = (brightness * 0.3 + contrast * 0.3 + 
                   sharpness * 0.25 + resolution_score * 0.15)

        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'resolution_score': float(resolution_score),
            'overall': float(overall)
        }

    def needs_enhancement(self, quality: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if image needs enhancement.
        
        Returns: (needs_enhancement, reason)
        """
        if quality['brightness'] < self.quality_thresholds['brightness']['low']:
            return True, 'low_brightness'
        
        if quality['brightness'] > self.quality_thresholds['brightness']['high']:
            return False, 'brightness_ok'
        
        if quality['contrast'] < self.quality_thresholds['contrast']['low']:
            return True, 'low_contrast'
        
        if quality['sharpness'] < self.quality_thresholds['sharpness']['low']:
            return True, 'low_sharpness'
        
        return False, 'quality_ok'

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Good for improving contrast in uneven lighting.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l = self.clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def apply_histogram_eq(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization.
        Good for low-light conditions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = equalized
        
        return enhanced

    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction for brightness adjustment.
        gamma > 1 brightens, gamma < 1 darkens
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Auto-select and apply enhancement based on quality assessment.
        
        Returns: (enhanced_image, method_used)
        """
        quality = self.assess_quality(image)
        needs, reason = self.needs_enhancement(quality)
        
        if not needs:
            return image, 'none'
        
        enhanced = image.copy()
        method = 'none'
        
        if reason == 'low_brightness':
            if quality['contrast'] < 0.4:
                enhanced = self.apply_histogram_eq(image)
                method = 'histogram_eq'
            else:
                enhanced = self.apply_gamma_correction(image, gamma=1.3)
                method = 'gamma'
        
        elif reason == 'low_contrast':
            enhanced = self.apply_clahe(image)
            method = 'clahe'
        
        elif reason == 'low_sharpness':
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(image, -1, kernel)
            method = 'sharpen'
        
        if method == 'none':
            if quality['brightness'] < 0.4:
                enhanced = self.apply_gamma_correction(image, gamma=1.2)
                method = 'gamma'
            elif quality['contrast'] < 0.4:
                enhanced = self.apply_clahe(image)
                method = 'clahe'
        
        return enhanced, method

    def visualize_preprocessing(self, original: np.ndarray, enhanced: np.ndarray, 
                                original_quality: Dict, enhanced_quality: Dict,
                                method: str) -> np.ndarray:
        """
        Generate side-by-side comparison visualization.
        """
        h, w = original.shape[:2]
        
        viz = np.zeros((h + 80, w * 2, 3), dtype=np.uint8)
        viz.fill(30)
        
        viz[0:h, 0:w] = original
        viz[0:h, w:w*2] = enhanced
        
        label_h = h + 10
        cv2.putText(viz, f"Original (score: {original_quality.get('overall', 0):.2f})", 
                   (10, label_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(viz, f"Enhanced: {method} (score: {enhanced_quality.get('overall', 0):.2f})", 
                   (w + 10, label_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        bar_y = h + 30
        bar_height = 10
        max_bar = w // 2 - 20
        start_x = 10
        start_x2 = w + 10
        
        metrics = ['brightness', 'contrast', 'sharpness']
        colors = [(255, 193, 7), (255, 152, 0), (255, 87, 34)]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            orig_val = original_quality.get(metric, 0)
            enh_val = enhanced_quality.get(metric, 0)
            
            y = bar_y + i * (bar_height + 5)
            
            cv2.rectangle(viz, (start_x, y), (start_x + int(orig_val * max_bar), y + bar_height), color, -1)
            cv2.rectangle(viz, (start_x, y), (start_x + max_bar, y + bar_height), (60, 60, 60), 1)
            
            cv2.rectangle(viz, (start_x2, y), (start_x2 + int(enh_val * max_bar), y + bar_height), color, -1)
            cv2.rectangle(viz, (start_x2, y), (start_x2 + max_bar, y + bar_height), (60, 60, 60), 1)
        
        return viz

    def get_preprocessing_info(self, original: np.ndarray, enhanced: np.ndarray, 
                               method: str) -> Dict:
        """
        Get detailed preprocessing information for API response.
        """
        original_quality = self.assess_quality(original)
        enhanced_quality = self.assess_quality(enhanced)
        
        return {
            'was_enhanced': method != 'none',
            'method': method,
            'original_quality': original_quality,
            'enhanced_quality': enhanced_quality,
            'improvement': {
                'brightness': enhanced_quality['brightness'] - original_quality['brightness'],
                'contrast': enhanced_quality['contrast'] - original_quality['contrast'],
                'sharpness': enhanced_quality['sharpness'] - original_quality['sharpness'],
                'overall': enhanced_quality['overall'] - original_quality['overall']
            }
        }


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to preprocess an image.
    
    Returns: (enhanced_image, preprocessing_info)
    """
    preprocessor = ImagePreprocessor()
    
    quality = preprocessor.assess_quality(image)
    needs, reason = preprocessor.needs_enhancement(quality)
    
    if needs:
        enhanced, method = preprocessor.enhance(image)
    else:
        enhanced = image
        method = 'none'
    
    info = preprocessor.get_preprocessing_info(image, enhanced, method)
    
    return enhanced, info


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    
    test_images = [
        'test_images/test_subject.jpg',
        '_examples/reference_images/kanye_west_ref.jpg'
    ]
    
    for path in test_images:
        import os
        if os.path.exists(path):
            img = cv2.imread(path)
            print(f"\n=== Testing: {path} ===")
            
            quality = preprocessor.assess_quality(img)
            print(f"Original quality: {quality}")
            
            needs, reason = preprocessor.needs_enhancement(quality)
            print(f"Needs enhancement: {needs} ({reason})")
            
            enhanced, method = preprocessor.enhance(img)
            print(f"Enhancement method: {method}")
            
            enhanced_quality = preprocessor.assess_quality(enhanced)
            print(f"Enhanced quality: {enhanced_quality}")
            
            break
