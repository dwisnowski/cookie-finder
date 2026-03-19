"""
Thermal frame processor with stateful detection and transformation pipeline.
Designed for both standalone (OpenCV GUI) and web server (MJPEG) modes.

This class encapsulates all frame processing logic to enable code sharing
between different output frontends (GUI, web, etc.).
"""

import cv2
import numpy as np


# Static thermal color palettes
THERMAL_PALETTES = [
    ("Ironbow", cv2.COLORMAP_INFERNO),
    ("Rainbow", cv2.COLORMAP_JET),
    ("Lava", cv2.COLORMAP_HOT),
    ("Ocean", cv2.COLORMAP_VIRIDIS),
    ("Magma", cv2.COLORMAP_MAGMA),
    ("WhiteHot", cv2.COLORMAP_BONE),
    ("BlackHot", cv2.COLORMAP_TWILIGHT),
]


class ThermalProcessor:
    """Stateful thermal frame processor."""
    
    def __init__(self):
        """Initialize processor with default state."""
        
        # Detection modes
        self.heat_seeker_mode = False
        self.heat_cluster_mode = False
        self.motion_mode = False
        self.upscale_mode = False
        self.superres_mode = False
        self.openvino_sr_mode = False
        self.palette_mode = False
        self.denoise_mode = False
        self.normalize_mode = False
        self.enhance_mode = False
        self.stabilize_mode = False
        self.stabilize_super = False
        self.threshold_mode = False
        self.show_text = True
        self.yolo_mode = False
        self.optical_flow_mode = False
        self.optical_flow_masked_mode = False
        self.isotherm_mode = False
        
        # Mode-specific parameters
        self.palette_idx = 0
        self.threshold_value = 127
        self.optical_flow_threshold = 100
        self.isotherm_min = 100
        self.isotherm_max = 200
        self.isotherm_use_black = False
        self.heat_seeker_max_boxes = 5
        self.heat_seeker_min_brightness = 50
        self.yolo_skip_frames = False
        
        # Stabilization parameters
        self.stabilize_use_orb = True
        self.stabilize_strength = 1.0
        self.stabilize_smooth = 0.7
        self.stabilize_show_points = False
        self.phase_strength = 1.0
        self.phase_buffer_size = 5
        self.orb_buffer_size = 1
        
        # ORB stabilizer state
        self.orb_detector = cv2.ORB_create(nfeatures=2000)
        self.orb_prev_gray = None
        self.orb_prev_kps = None
        self.orb_prev_des = None
        self.orb_smoothed_matrix = np.eye(2, 3, dtype=np.float32)
        self.orb_matrix_buffer = []
        self.orb_ref_gray = None
        self.orb_ref_kps = None
        self.orb_ref_des = None
        self.orb_frame_count = 0
        self.orb_matched_kps = []
        
        # Phase correlation state
        self.phase_prev_gray = None
        self.phase_shift_buffer = []
        self.phase_ref_gray = None
        self.phase_frame_count = 0
        
        # Super-resolution state
        self.superres_frame_buffer = []
        self.superres_buffer_size = 5
        
        # Denoise state
        self.accumulated_frame = None
        self.denoise_alpha = 0.2
        
        # YOLO model (lazy loaded)
        self.yolo_model = None
        self.yolo_result_frame = None
        self.frame_count = 0
        
        # OpenVINO availability
        self.openvino_available = False
        try:
            from openvino.runtime import Core
            self.openvino_available = True
        except ImportError:
            self.openvino_available = False
    
    # ========== Getters/Setters ==========
    
    def get_state(self):
        """Return dictionary of all current state (modes and parameters)."""
        return {
            # Modes
            'heat_seeker_mode': self.heat_seeker_mode,
            'heat_cluster_mode': self.heat_cluster_mode,
            'motion_mode': self.motion_mode,
            'upscale_mode': self.upscale_mode,
            'superres_mode': self.superres_mode,
            'openvino_sr_mode': self.openvino_sr_mode,
            'palette_mode': self.palette_mode,
            'denoise_mode': self.denoise_mode,
            'normalize_mode': self.normalize_mode,
            'enhance_mode': self.enhance_mode,
            'stabilize_mode': self.stabilize_mode,
            'stabilize_super': self.stabilize_super,
            'threshold_mode': self.threshold_mode,
            'show_text': self.show_text,
            'yolo_mode': self.yolo_mode,
            'optical_flow_mode': self.optical_flow_mode,
            'optical_flow_masked_mode': self.optical_flow_masked_mode,
            'isotherm_mode': self.isotherm_mode,
            # Parameters
            'palette_idx': self.palette_idx,
            'palette_name': THERMAL_PALETTES[self.palette_idx][0] if self.palette_idx < len(THERMAL_PALETTES) else "Unknown",
            'threshold_value': self.threshold_value,
            'optical_flow_threshold': self.optical_flow_threshold,
            'isotherm_min': self.isotherm_min,
            'isotherm_max': self.isotherm_max,
            'heat_seeker_max_boxes': self.heat_seeker_max_boxes,
            'heat_seeker_min_brightness': self.heat_seeker_min_brightness,
            'stabilize_strength': self.stabilize_strength,
            'stabilize_smooth': self.stabilize_smooth,
            'phase_strength': self.phase_strength,
        }
    
    def set_mode(self, mode_name, enabled):
        """Toggle a detection mode on/off. Mutually exclusive modes turn off others."""
        exclusive_modes = [
            'heat_seeker_mode', 'heat_cluster_mode', 'motion_mode',
            'palette_mode', 'threshold_mode', 'yolo_mode',
            'optical_flow_mode', 'optical_flow_masked_mode', 'isotherm_mode'
        ]
        
        if mode_name in exclusive_modes and enabled:
            # Turn off other exclusive modes
            for m in exclusive_modes:
                if m != mode_name:
                    setattr(self, m, False)
        
        setattr(self, mode_name, enabled)
    
    def set_parameter(self, param_name, value):
        """Set a parameter value with appropriate bounds checking."""
        if param_name == 'palette_idx':
            self.palette_idx = int(value) % len(THERMAL_PALETTES)
        elif param_name == 'threshold_value':
            self.threshold_value = max(0, min(255, int(value)))
        elif param_name == 'optical_flow_threshold':
            self.optical_flow_threshold = max(0, min(255, int(value)))
        elif param_name == 'isotherm_min':
            self.isotherm_min = max(0, min(self.isotherm_max, int(value)))
        elif param_name == 'isotherm_max':
            self.isotherm_max = max(self.isotherm_min, min(255, int(value)))
        elif param_name == 'heat_seeker_max_boxes':
            self.heat_seeker_max_boxes = max(1, min(15, int(value)))
        elif param_name == 'heat_seeker_min_brightness':
            self.heat_seeker_min_brightness = max(0, min(255, int(value)))
        elif param_name == 'stabilize_strength':
            self.stabilize_strength = max(0.0, min(1.0, float(value)))
        elif param_name == 'stabilize_smooth':
            self.stabilize_smooth = max(0.0, min(0.99, float(value)))
        elif param_name == 'phase_strength':
            self.phase_strength = max(0.1, min(3.0, float(value)))
        elif param_name == 'phase_buffer_size':
            self.phase_buffer_size = max(1, min(20, int(value)))
        elif param_name == 'orb_buffer_size':
            self.orb_buffer_size = max(1, min(20, int(value)))
        else:
            setattr(self, param_name, value)
    
    # ========== Utility Functions (Stateless) ==========
    
    @staticmethod
    def detect_hot_spots(frame, threshold_percentile=85):
        """Detect brightest pixel clusters and return contours."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, threshold_percentile)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def detect_heat_clusters(frame, threshold_percentile=80, min_area=200):
        """Detect largest heat clusters using brightness clustering."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, threshold_percentile)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        return valid_contours[:5]
    
    @staticmethod
    def detect_motion_objects(frame, prev_frame):
        """Detect moving objects using frame differencing."""
        if prev_frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        return valid_contours[:5]
    
    @staticmethod
    def draw_heat_boxes(frame, contours, min_area=100, max_boxes=None, min_brightness=0):
        """Draw bounding boxes around hot spots."""
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if max_boxes is not None:
            sorted_contours = sorted_contours[:max_boxes]
        
        boxes_drawn = 0
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                max_brightness = np.max(roi)
                
                if max_brightness >= min_brightness:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    boxes_drawn += 1
        
        return output, boxes_drawn
    
    @staticmethod
    def draw_cluster_boxes(frame, contours):
        """Draw bounding boxes around heat clusters with labels."""
        output = frame.copy()
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label = f"Cluster {idx + 1}: {int(area)}"
            cv2.putText(output, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return output
    
    @staticmethod
    def draw_motion_boxes(frame, contours):
        """Draw bounding boxes around detected motion."""
        output = frame.copy()
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = f"Motion {idx + 1}"
            cv2.putText(output, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return output
    
    @staticmethod
    def apply_thermal_palette(frame, palette_idx):
        """Apply thermal color palette to frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, colormap = THERMAL_PALETTES[palette_idx % len(THERMAL_PALETTES)]
        thermal_frame = cv2.applyColorMap(gray, colormap)
        return thermal_frame
    
    @staticmethod
    def threshold_frame(frame, threshold_value=127):
        """Apply thresholding to isolate interest items."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        return thresholded_bgr
    
    @staticmethod
    def isotherm_highlight(frame, min_threshold=100, max_threshold=200, use_black=False):
        """Highlight specific heat range (isotherm) in red or black."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = (gray >= min_threshold) & (gray <= max_threshold)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        if use_black:
            output[mask] = [0, 0, 0]
        else:
            output[mask] = [0, 0, 255]
        
        return output
    
    @staticmethod
    def normalize_frame(frame):
        """Normalize frame to stretch 0-255 range."""
        try:
            frame_umat = cv2.UMat(frame)
            frame_float = cv2.UMat(frame_umat, cv2.CV_32F)
            min_val = cv2.minMaxLoc(frame_float)[0]
            max_val = cv2.minMaxLoc(frame_float)[1]
            
            if max_val == min_val:
                return frame
            
            range_val = max_val - min_val
            normalized = (frame_float - min_val) / range_val * 255.0
            normalized = cv2.convertScaleAbs(normalized)
            return normalized.get()
        except:
            # Fallback to CPU
            frame_float = frame.astype(np.float32)
            min_val = np.min(frame_float)
            max_val = np.max(frame_float)
            
            if max_val == min_val:
                return frame
            
            range_val = max_val - min_val
            normalized = (frame_float - min_val) / range_val * 255.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            return normalized
    
    @staticmethod
    def enhance_details(frame):
        """Apply CLAHE to pull out subtle thermal textures."""
        try:
            frame_umat = cv2.UMat(frame)
            gray_umat = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray_umat)
            result = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            return result.get()
        except:
            # Fallback to CPU
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def upscale_frame(frame, scale=2):
        """Upscale frame using Lanczos interpolation."""
        h, w = frame.shape[:2]
        new_w = w * scale
        new_h = h * scale
        upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return upscaled
    
    @staticmethod
    def calculate_optical_flow(frame, prev_frame):
        """Calculate optical flow to visualize heat velocity."""
        if prev_frame is None:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        
        flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_frame
    
    @staticmethod
    def calculate_optical_flow_masked(frame, prev_frame, threshold=100):
        """Calculate optical flow only where heat signature exceeds threshold."""
        if prev_frame is None:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        small_prev = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
        
        flow = cv2.calcOpticalFlowFarneback(small_prev, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv = np.zeros((small_gray.shape[0], small_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_final = cv2.resize(flow_bgr, (frame.shape[1], frame.shape[0]))
        
        mask = gray > threshold
        result = frame.copy()
        result[mask] = cv2.addWeighted(frame[mask], 0.5, flow_final[mask], 0.5, 0)
        return result
    
    # ========== Stateful Methods ==========
    
    def apply_denoise(self, frame, alpha=0.2):
        """Apply temporal accumulation denoising."""
        if self.accumulated_frame is None:
            self.accumulated_frame = frame.copy()
            return frame
        
        denoised = cv2.addWeighted(frame, alpha, self.accumulated_frame, 1 - alpha, 0)
        self.accumulated_frame = denoised.copy()
        return denoised
    
    def stabilize_frame_orb(self, frame, strength=1.0):
        """Stabilize frame using ORB feature tracking + RANSAC."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb_detector.detectAndCompute(gray, None)
        
        if self.orb_ref_gray is None or des is None:
            self.orb_ref_gray, self.orb_ref_kps, self.orb_ref_des = gray, kps, des
            self.orb_prev_gray, self.orb_prev_kps, self.orb_prev_des = gray, kps, des
            self.orb_matched_kps = []
            return frame
        
        self.orb_frame_count += 1
        if self.orb_frame_count > 30:
            self.orb_ref_gray, self.orb_ref_kps, self.orb_ref_des = gray, kps, des
            self.orb_frame_count = 0
        
        try:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.orb_ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)[:200]
            
            if len(matches) < 4:
                self.orb_matched_kps = []
                return frame
            
            pts1 = np.float32([self.orb_ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            self.orb_matched_kps = [tuple(map(int, pt[0])) for pt in pts2]
            
            matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=10.0)
            
            if matrix is not None:
                self.orb_matrix_buffer.append(matrix.copy())
                if len(self.orb_matrix_buffer) > self.orb_buffer_size:
                    self.orb_matrix_buffer.pop(0)
                
                avg_matrix = np.mean(self.orb_matrix_buffer, axis=0)
                self.orb_smoothed_matrix = self.stabilize_smooth * self.orb_smoothed_matrix + (1 - self.stabilize_smooth) * avg_matrix
                
                identity = np.eye(2, 3, dtype=np.float32)
                blended_matrix = identity + (self.orb_smoothed_matrix - identity) * strength
                
                stabilized = cv2.warpAffine(frame, blended_matrix, (frame.shape[1], frame.shape[0]))
                return stabilized
            
            return frame
        except:
            self.orb_matched_kps = []
            return frame
    
    def stabilize_frame_phase_correlation(self, frame):
        """Super stabilization using phase correlation with reference frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if self.phase_ref_gray is None:
            self.phase_ref_gray = gray
            self.phase_prev_gray = gray
            return frame
        
        self.phase_frame_count += 1
        if self.phase_frame_count > 60:
            self.phase_ref_gray = gray
            self.phase_frame_count = 0
        
        try:
            h, w = gray.shape
            window = np.outer(np.hanning(h), np.hanning(w))
            windowed_ref = self.phase_ref_gray * window
            windowed_gray = gray * window
            
            shift, response = cv2.phaseCorrelate(windowed_ref, windowed_gray)
            dx, dy = shift
            
            if response < 0.1:
                return frame
            
            self.phase_shift_buffer.append((dx, dy))
            if len(self.phase_shift_buffer) > self.phase_buffer_size:
                self.phase_shift_buffer.pop(0)
            
            avg_dx = np.mean([s[0] for s in self.phase_shift_buffer])
            avg_dy = np.mean([s[1] for s in self.phase_shift_buffer])
            
            avg_dx = -avg_dx * self.phase_strength
            avg_dy = -avg_dy * self.phase_strength
            
            avg_dx = np.clip(avg_dx, -50, 50)
            avg_dy = np.clip(avg_dy, -50, 50)
            
            matrix = np.float32([[1, 0, avg_dx], [0, 1, avg_dy]])
            stabilized = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
            
            self.phase_prev_gray = gray
            return stabilized
        except Exception:
            self.phase_prev_gray = gray
            return frame
    
    def apply_superres(self, frame):
        """Apply temporal super-resolution using multi-frame reconstruction."""
        try:
            self.superres_frame_buffer.append(frame.copy())
            if len(self.superres_frame_buffer) > self.superres_buffer_size:
                self.superres_frame_buffer.pop(0)
            
            if len(self.superres_frame_buffer) < 3:
                return frame
            
            avg_frame = np.mean(self.superres_frame_buffer, axis=0).astype(np.uint8)
            upscaled = cv2.resize(avg_frame, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            return upscaled
        except Exception:
            return frame
    
    def init_yolo_model(self):
        """Initialize YOLOv8 Nano model (lazy loaded)."""
        if self.yolo_model is not None:
            return self.yolo_model
        
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            return self.yolo_model
        except ImportError:
            print("Error: ultralytics not installed. Run: make install-yolo")
            return None
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    
    def detect_with_yolo(self, frame):
        """Run YOLO detection on frame."""
        if self.yolo_model is None:
            return frame
        
        try:
            results = self.yolo_model.track(frame, persist=True, stream=True, 
                                           verbose=False, imgsz=320, device='cpu')
            annotated_frame = frame.copy()
            
            for r in results:
                annotated_frame = r.plot()
            
            return annotated_frame
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return frame
    
    # ========== Main Processing Orchestrator ==========
    
    def process_frame(self, frame, prev_frame=None):
        """
        Process frame according to current mode settings.
        
        Args:
            frame: Input BGR frame from camera
            prev_frame: Previous frame (optional, for motion detection/optical flow)
        
        Returns:
            Tuple of (processed_frame, mode_text, metadata_dict)
        """
        display_frame = frame.copy()
        mode_text = ""
        metadata = {}
        
        self.frame_count += 1
        
        # Apply preprocessing (normalize, enhance)
        if self.normalize_mode:
            display_frame = self.normalize_frame(display_frame)
        
        if self.enhance_mode:
            display_frame = self.enhance_details(display_frame)
        
        # Apply stabilization
        if self.stabilize_mode:
            if self.stabilize_use_orb:
                display_frame = self.stabilize_frame_orb(display_frame, strength=self.stabilize_strength)
            else:
                if prev_frame is not None:
                    try:
                        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        warp_mode = cv2.MOTION_TRANSLATION
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
                        _, warp_matrix = cv2.findTransformECC(prev_gray, gray, warp_matrix, warp_mode, criteria)
                        display_frame = cv2.warpAffine(display_frame, warp_matrix, 
                                                       (display_frame.shape[1], display_frame.shape[0]),
                                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    except:
                        pass
        
        if self.stabilize_super:
            display_frame = self.stabilize_frame_phase_correlation(display_frame)
        
        # Apply denoising
        if self.denoise_mode:
            display_frame = self.apply_denoise(display_frame, alpha=self.denoise_alpha)
        
        # Apply main processing mode
        if self.threshold_mode:
            display_frame = self.threshold_frame(display_frame, self.threshold_value)
            mode_text = f"Threshold: {self.threshold_value}"
        
        elif self.palette_mode:
            display_frame = self.apply_thermal_palette(display_frame, self.palette_idx)
            palette_name = THERMAL_PALETTES[self.palette_idx][0]
            mode_text = f"Palette: {palette_name}"
        
        elif self.heat_seeker_mode:
            contours = self.detect_hot_spots(frame)
            display_frame, boxes_drawn = self.draw_heat_boxes(
                display_frame, contours, max_boxes=self.heat_seeker_max_boxes,
                min_brightness=self.heat_seeker_min_brightness
            )
            metadata['boxes_drawn'] = boxes_drawn
            mode_text = f"Heat-Seeker: ON (Max: {self.heat_seeker_max_boxes}, Min brightness: {self.heat_seeker_min_brightness}, Drawn: {boxes_drawn})"
        
        elif self.heat_cluster_mode:
            contours = self.detect_heat_clusters(frame)
            display_frame = self.draw_cluster_boxes(display_frame, contours)
            mode_text = "Heat-Cluster: ON"
        
        elif self.motion_mode:
            if prev_frame is not None:
                contours = self.detect_motion_objects(frame, prev_frame)
                display_frame = self.draw_motion_boxes(display_frame, contours)
            mode_text = "Motion Detection: ON"
        
        elif self.yolo_mode:
            if self.yolo_model is None:
                self.init_yolo_model()
            
            if self.yolo_model is not None:
                if self.yolo_skip_frames:
                    if self.frame_count % 4 == 0:
                        self.yolo_result_frame = self.detect_with_yolo(display_frame)
                    
                    if self.yolo_result_frame is not None:
                        display_frame = self.yolo_result_frame
                    mode_text = "YOLO AI Detection: ON (Skip-Frame)"
                else:
                    display_frame = self.detect_with_yolo(display_frame)
                    mode_text = "YOLO AI Detection: ON"
        
        elif self.optical_flow_mode:
            display_frame = self.calculate_optical_flow(display_frame, prev_frame)
            mode_text = "Optical Flow: ON"
        
        elif self.optical_flow_masked_mode:
            display_frame = self.calculate_optical_flow_masked(display_frame, prev_frame, self.optical_flow_threshold)
            mode_text = f"Optical Flow (Masked, threshold={self.optical_flow_threshold}): ON"
        
        elif self.isotherm_mode:
            display_frame = self.isotherm_highlight(display_frame, self.isotherm_min, 
                                                   self.isotherm_max, self.isotherm_use_black)
            mask_color = "Black" if self.isotherm_use_black else "Red"
            mode_text = f"Isotherm ({mask_color}): {self.isotherm_min}-{self.isotherm_max}"
        
        else:
            mode_text = "Normal View"
        
        # Apply upscaling/super-resolution
        if self.upscale_mode:
            display_frame = self.upscale_frame(display_frame, scale=2)
        
        if self.superres_mode:
            display_frame = self.apply_superres(display_frame)
        
        if self.openvino_sr_mode and self.openvino_available:
            display_frame = self.upscale_frame(display_frame, scale=2)  # Placeholder
        
        # Build text description
        text_lines = [mode_text]
        
        if self.denoise_mode:
            text_lines.append("Denoise: ON")
        if self.normalize_mode:
            text_lines.append("Normalize: ON")
        if self.enhance_mode:
            text_lines.append("Enhance: ON")
        if self.stabilize_mode:
            text_lines.append(f"Stabilize: ON (Strength: {self.stabilize_strength:.1f}, Smooth: {self.stabilize_smooth:.2f}, Buffer: {self.orb_buffer_size})")
        if self.stabilize_super:
            text_lines.append(f"Super Stabilization: ON (Strength: {self.phase_strength:.1f}, Buffer: {self.phase_buffer_size})")
        if self.upscale_mode:
            text_lines.append("Upscale: ON")
        if self.superres_mode:
            text_lines.append("Super-Resolution: ON")
        if self.openvino_sr_mode:
            text_lines.append("OpenVINO Super-Resolution: ON")
        
        metadata['text_lines'] = text_lines
        
        return display_frame, mode_text, metadata
