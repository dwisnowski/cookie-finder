import cv2
import sys
import numpy as np

# Thermal color palettes
THERMAL_PALETTES = [
    ("Ironbow", cv2.COLORMAP_INFERNO),
    ("Rainbow", cv2.COLORMAP_JET),
    ("Lava", cv2.COLORMAP_HOT),
    ("Ocean", cv2.COLORMAP_VIRIDIS),
    ("Magma", cv2.COLORMAP_MAGMA),
    ("WhiteHot", cv2.COLORMAP_BONE),
    ("BlackHot", cv2.COLORMAP_TWILIGHT),
]

# Initialize YOLO model (lazy load)
yolo_model = None

def detect_hot_spots(frame, threshold_percentile=85):
    """Detect brightest pixel clusters and return bounding boxes."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate threshold based on percentile
    threshold = np.percentile(gray, threshold_percentile)
    
    # Create binary mask of hot spots
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to connect nearby hot spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def detect_heat_clusters(frame, threshold_percentile=80, min_area=200):
    """Detect largest heat clusters using brightness clustering."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get white/bright pixels
    threshold = np.percentile(gray, threshold_percentile)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up and connect clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and sort by size
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    return valid_contours[:5]  # Return top 5 largest clusters

def detect_motion_objects(frame, prev_frame):
    """Detect moving objects using frame differencing."""
    if prev_frame is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(gray, prev_gray)
    
    # Threshold to get motion areas
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    return valid_contours[:5]

def draw_heat_boxes(frame, contours, min_area=100, max_boxes=None, min_brightness=0):
    """Draw bounding boxes around hot spots, limited to max_boxes and min brightness."""
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sort contours by area (largest first)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Limit number of boxes if specified
    if max_boxes is not None:
        sorted_contours = sorted_contours[:max_boxes]
    
    boxes_drawn = 0
    for contour in sorted_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if box contains pixels above min_brightness threshold
            roi = gray[y:y+h, x:x+w]
            max_brightness = np.max(roi)
            
            if max_brightness >= min_brightness:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                boxes_drawn += 1
    
    return output, boxes_drawn

def draw_cluster_boxes(frame, contours):
    """Draw bounding boxes around heat clusters with labels."""
    output = frame.copy()
    
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Draw label with area
        label = f"Cluster {idx + 1}: {int(area)}"
        cv2.putText(output, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return output

def draw_motion_boxes(frame, contours):
    """Draw bounding boxes around detected motion."""
    output = frame.copy()
    
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw label
        label = f"Motion {idx + 1}"
        cv2.putText(output, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return output

def upscale_frame(frame, scale=2):
    """Upscale frame using Lanczos interpolation."""
    h, w = frame.shape[:2]
    new_w = w * scale
    new_h = h * scale
    upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return upscaled

def apply_thermal_palette(frame, palette_idx):
    """Apply thermal color palette to frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, colormap = THERMAL_PALETTES[palette_idx]
    thermal_frame = cv2.applyColorMap(gray, colormap)
    return thermal_frame

def denoise_frame(frame, accumulated_frame, alpha=0.2):
    """Apply temporal accumulation denoising using frame averaging."""
    if accumulated_frame is None:
        return frame, frame.copy()
    denoised = cv2.addWeighted(frame, alpha, accumulated_frame, 1 - alpha, 0)
    return denoised, denoised.copy()

def normalize_frame(frame):
    """Normalize frame to stretch 0-255 range."""
    normalized = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized

def enhance_details(frame):
    """Apply CLAHE to pull out subtle thermal textures."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# Global ORB stabilizer state
orb_detector = cv2.ORB_create(nfeatures=2000)
orb_prev_gray = None
orb_prev_kps = None
orb_prev_des = None
orb_stabilization_strength = 1.0
orb_smoothed_matrix = np.eye(2, 3, dtype=np.float32)
orb_smooth_factor = 0.7
orb_matrix_buffer = []  # Buffer to store transformation matrices
orb_buffer_size = 1  # Number of frames to average (1 = no buffering)
orb_ref_gray = None  # Reference frame for stabilization
orb_ref_kps = None
orb_ref_des = None
orb_frame_count = 0  # Counter to refresh reference frame periodically

def stabilize_frame_orb(frame, strength=1.0):
    """Stabilize frame using ORB feature tracking + RANSAC with reference frame approach."""
    global orb_prev_gray, orb_prev_kps, orb_prev_des, orb_smoothed_matrix, orb_smooth_factor, orb_matrix_buffer, orb_buffer_size
    global orb_ref_gray, orb_ref_kps, orb_ref_des, orb_frame_count
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps, des = orb_detector.detectAndCompute(gray, None)
    
    # Initialize reference frame on first call
    if orb_ref_gray is None or des is None:
        orb_ref_gray, orb_ref_kps, orb_ref_des = gray, kps, des
        orb_prev_gray, orb_prev_kps, orb_prev_des = gray, kps, des
        return frame
    
    # Refresh reference frame every 30 frames to prevent drift
    orb_frame_count += 1
    if orb_frame_count > 30:
        orb_ref_gray, orb_ref_kps, orb_ref_des = gray, kps, des
        orb_frame_count = 0
    
    try:
        # Match features against reference frame
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(orb_ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:200]
        
        if len(matches) < 4:
            return frame
        
        # Extract coordinates
        pts1 = np.float32([orb_ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find rigid transformation (translation + rotation) with RANSAC
        matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=10.0)
        
        if matrix is not None:
            # Add matrix to buffer
            orb_matrix_buffer.append(matrix.copy())
            
            # Keep buffer size limited
            if len(orb_matrix_buffer) > orb_buffer_size:
                orb_matrix_buffer.pop(0)
            
            # Average all matrices in buffer
            avg_matrix = np.mean(orb_matrix_buffer, axis=0)
            
            # Temporal smoothing: blend averaged matrix with previous smoothed matrix
            orb_smoothed_matrix = orb_smooth_factor * orb_smoothed_matrix + (1 - orb_smooth_factor) * avg_matrix
            
            # Apply strength factor to the smoothed transformation
            identity = np.eye(2, 3, dtype=np.float32)
            blended_matrix = identity + (orb_smoothed_matrix - identity) * strength
            
            # Warp current frame to align with reference
            stabilized = cv2.warpAffine(frame, blended_matrix, (frame.shape[1], frame.shape[0]))
            return stabilized
        
        return frame
    except:
        return frame

def stabilize_frame(frame, prev_frame):
    """Stabilize frame using ECC (Enhanced Correlation Coefficient)."""
    if prev_frame is None:
        return frame
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Define motion model (translation only is fastest)
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Find transformation (limit iterations for 50 FPS speed)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
        _, warp_matrix = cv2.findTransformECC(prev_gray, gray, warp_matrix, warp_mode, criteria)
        
        # Apply transformation to current frame
        stabilized = cv2.warpAffine(frame, warp_matrix, (frame.shape[1], frame.shape[0]),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return stabilized
    except:
        return frame  # Fallback if alignment fails

def threshold_frame(frame, threshold_value=127):
    """Apply thresholding to isolate interest items based on brightness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    # Convert back to BGR for consistency
    thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    return thresholded_bgr

def init_yolo_model():
    """Initialize YOLOv8 Nano model."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("YOLOv8 Nano model loaded successfully")
        return model
    except ImportError:
        print("Error: ultralytics not installed. Run: make install")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def detect_with_yolo(frame, model):
    """Run YOLO detection on frame with Intel optimization."""
    if model is None:
        return frame
    
    try:
        # Use imgsz=320 for Intel speed (~70% faster than default 640)
        results = model.track(frame, persist=True, stream=True, verbose=False, imgsz=320, device='cpu')
        annotated_frame = frame.copy()
        
        for r in results:
            annotated_frame = r.plot()
        
        return annotated_frame
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return frame

def calculate_optical_flow(frame, prev_frame):
    """Calculate optical flow to visualize heat velocity."""
    if prev_frame is None:
        return frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Create visualization
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    
    flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_frame

def calculate_optical_flow_masked(frame, prev_frame, threshold=100):
    """Calculate optical flow only where heat signature exceeds threshold."""
    if prev_frame is None:
        return frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Low-res flow calculation (Step 1 for speed on Intel)
    # Reducing the scale here significantly improves FPS
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    small_prev = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
    
    flow = cv2.calcOpticalFlowFarneback(small_prev, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 2. Visualization
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((small_gray.shape[0], small_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 3. Resize back and mask
    flow_final = cv2.resize(flow_bgr, (frame.shape[1], frame.shape[0]))
    mask = gray > threshold
    
    # Only show flow where it's 'hot'
    result = frame.copy()
    result[mask] = cv2.addWeighted(frame[mask], 0.5, flow_final[mask], 0.5, 0)
    
    return result

def isotherm_highlight(frame, min_threshold=100, max_threshold=200, use_black=False):
    """Highlight specific heat range (isotherm) in red or black, rest in grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create mask for pixels within the threshold range
    mask = (gray >= min_threshold) & (gray <= max_threshold)
    
    # Create output frame (grayscale)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Highlight masked region in red or black
    if use_black:
        output[mask] = [0, 0, 0]  # Black in BGR
    else:
        output[mask] = [0, 0, 255]  # Red in BGR
    
    return output
def main():
    global yolo_model, orb_buffer_size
    
    # Try to open the thermal camera (usually device 0, adjust if needed)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera. Check USB connection.")
        sys.exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    heat_seeker_mode = False
    heat_cluster_mode = False
    motion_mode = False
    upscale_mode = False
    palette_mode = False
    denoise_mode = False
    normalize_mode = False
    enhance_mode = False
    stabilize_mode = False
    stabilize_use_orb = True
    stabilize_strength = 1.0
    stabilize_smooth = 0.7
    threshold_mode = False
    yolo_mode = False
    optical_flow_mode = False
    optical_flow_masked_mode = False
    isotherm_mode = False
    yolo_skip_frames = False
    frame_count = 0
    yolo_result_frame = None
    heat_seeker_max_boxes = 5
    heat_seeker_min_brightness = 50
    threshold_value = 127
    optical_flow_threshold = 100
    isotherm_min = 100
    isotherm_max = 200
    isotherm_use_black = False
    palette_idx = 0
    prev_frame = None
    accumulated_frame = None
    
    print("Thermal camera feed started.")
    print("Press 'h' to toggle Heat-Seeker mode")
    print("Press 'c' to toggle Heat-Cluster mode")
    print("Press 'm' to toggle Motion Detection mode")
    print("Press 'u' to toggle Upscale mode")
    print("Press 'p' to toggle Palette mode")
    print("Press 'd' to toggle Denoise mode")
    print("Press 'o' to toggle Normalize mode")
    print("Press 'e' to toggle Enhance Details (CLAHE) mode")
    print("Press 't' to toggle Threshold mode")
    print("Press 'y' to toggle YOLO AI Detection mode")
    print("Press 'shift+y' to toggle YOLO skip-frame optimization")
    print("Press 'f' to toggle Optical Flow mode")
    print("Press 'shift+f' to toggle Masked Optical Flow mode")
    print("Press 'i' to toggle Isotherm Highlight mode")
    print("Press '=' to increase threshold, '-' to decrease (in Threshold/Optical Flow modes)")
    print("Press arrow keys to adjust isotherm range (in Isotherm mode):")
    print("  Left arrow: decrease min threshold")
    print("  Right arrow: increase min threshold")
    print("  Down arrow: decrease max threshold")
    print("  Up arrow: increase max threshold")
    print("Press 'b' to toggle black/red mask (in Isotherm mode)")
    print("Press 'n' to cycle to next palette (in Palette mode)")
    print("Press 's' to toggle Stabilization mode")
    print("Press 'shift+s' to switch stabilization method (ORB/ECC)")
    print("Press '=' to increase stabilization strength (in Stabilization mode)")
    print("Press '-' to decrease stabilization strength (in Stabilization mode)")
    print("Press '[' to decrease temporal smoothing (in Stabilization mode)")
    print("Press ']' to increase temporal smoothing (in Stabilization mode)")
    print("Press '{' to decrease frame buffer size (in Stabilization mode)")
    print("Press '}' to increase frame buffer size (in Stabilization mode)")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        mode_text = ""
        
        # Apply Normalization if enabled
        if normalize_mode:
            display_frame = normalize_frame(display_frame)
        
        # Apply Enhance Details if enabled
        if enhance_mode:
            display_frame = enhance_details(display_frame)
        
        # Apply Stabilization if enabled
        if stabilize_mode:
            if stabilize_use_orb:
                # Update global smooth factor
                import sys
                sys.modules['__main__'].orb_smooth_factor = stabilize_smooth
                display_frame = stabilize_frame_orb(display_frame, strength=stabilize_strength)
            else:
                display_frame = stabilize_frame(display_frame, prev_frame)
        
        # Apply Denoise mode if enabled
        if denoise_mode:
            display_frame, accumulated_frame = denoise_frame(display_frame, accumulated_frame)
        
        # Apply Threshold mode if enabled
        if threshold_mode:
            display_frame = threshold_frame(display_frame, threshold_value)
            mode_text = f"Threshold: {threshold_value}"
        
        # Apply Palette mode if enabled
        elif palette_mode:
            display_frame = apply_thermal_palette(display_frame, palette_idx)
            palette_name, _ = THERMAL_PALETTES[palette_idx]
            mode_text = f"Palette: {palette_name}"
        
        # Apply Heat-Seeker mode if enabled
        elif heat_seeker_mode:
            contours = detect_hot_spots(frame)
            display_frame, boxes_drawn = draw_heat_boxes(display_frame, contours, max_boxes=heat_seeker_max_boxes, min_brightness=heat_seeker_min_brightness)
            mode_text = f"Heat-Seeker: ON (Max: {heat_seeker_max_boxes}, Min brightness: {heat_seeker_min_brightness}, Drawn: {boxes_drawn})"
        
        # Apply Heat-Cluster mode if enabled
        elif heat_cluster_mode:
            contours = detect_heat_clusters(frame)
            display_frame = draw_cluster_boxes(display_frame, contours)
            mode_text = "Heat-Cluster: ON"
        
        # Apply Motion Detection mode if enabled
        elif motion_mode:
            contours = detect_motion_objects(frame, prev_frame)
            display_frame = draw_motion_boxes(display_frame, contours)
            mode_text = "Motion Detection: ON"
        
        # If all off
        if not heat_seeker_mode and not heat_cluster_mode and not motion_mode and not palette_mode and not threshold_mode and not yolo_mode and not optical_flow_mode and not optical_flow_masked_mode and not isotherm_mode:
            mode_text = "Normal View"
        
        # Apply YOLO AI Detection if enabled
        if yolo_mode:
            if yolo_model is not None:
                # Skip-frame optimization for Intel Macs
                if yolo_skip_frames:
                    if frame_count % 4 == 0:  # Run AI every 4th frame (~12-15 FPS)
                        yolo_result_frame = detect_with_yolo(display_frame, yolo_model)
                    
                    # Use persistent rendering - show last result or raw frame
                    if yolo_result_frame is not None:
                        display_frame = yolo_result_frame
                    mode_text = "YOLO AI Detection: ON (Skip-Frame)"
                else:
                    display_frame = detect_with_yolo(display_frame, yolo_model)
                    mode_text = "YOLO AI Detection: ON"
        
        # Apply Optical Flow if enabled
        if optical_flow_mode:
            display_frame = calculate_optical_flow(display_frame, prev_frame)
            mode_text = "Optical Flow: ON"
        
        # Apply Masked Optical Flow if enabled
        if optical_flow_masked_mode:
            display_frame = calculate_optical_flow_masked(display_frame, prev_frame, optical_flow_threshold)
            mode_text = f"Optical Flow (Masked, threshold={optical_flow_threshold}): ON"
        
        # Apply Isotherm Highlight if enabled
        if isotherm_mode:
            display_frame = isotherm_highlight(display_frame, isotherm_min, isotherm_max, isotherm_use_black)
            mask_color = "Black" if isotherm_use_black else "Red"
            mode_text = f"Isotherm ({mask_color}): {isotherm_min}-{isotherm_max}"
        
        # Apply Upscale if enabled (works with all modes)
        if upscale_mode:
            display_frame = upscale_frame(display_frame, scale=2)
        
        # Add denoise indicator
        if denoise_mode:
            mode_text += " | Denoise: ON"
        
        # Add normalize indicator
        if normalize_mode:
            mode_text += " | Normalize: ON"
        
        # Add enhance indicator
        if enhance_mode:
            mode_text += " | Enhance: ON"
        
        # Add stabilize indicator
        if stabilize_mode:
            mode_text += f" | Stabilize: ON (Strength: {stabilize_strength:.1f}, Smooth: {stabilize_smooth:.2f}, Buffer: {orb_buffer_size})"
        
        # Add upscale indicator
        if upscale_mode:
            mode_text += " | Upscale: ON"
        
        # Add mode indicator
        cv2.putText(display_frame, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Thermal Camera Feed", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            heat_seeker_mode = not heat_seeker_mode
            heat_cluster_mode = False
            motion_mode = False
            status = "ON" if heat_seeker_mode else "OFF"
            print(f"Heat-Seeker mode: {status}")
        elif key == ord('c'):
            heat_cluster_mode = not heat_cluster_mode
            heat_seeker_mode = False
            motion_mode = False
            status = "ON" if heat_cluster_mode else "OFF"
            print(f"Heat-Cluster mode: {status}")
        elif key == ord('m'):
            motion_mode = not motion_mode
            heat_seeker_mode = False
            heat_cluster_mode = False
            status = "ON" if motion_mode else "OFF"
            print(f"Motion Detection mode: {status}")
        elif key == ord('u'):
            upscale_mode = not upscale_mode
            status = "ON" if upscale_mode else "OFF"
            print(f"Upscale mode: {status}")
        elif key == ord('p'):
            palette_mode = not palette_mode
            heat_seeker_mode = False
            heat_cluster_mode = False
            motion_mode = False
            status = "ON" if palette_mode else "OFF"
            print(f"Palette mode: {status}")
        elif key == ord('n') and palette_mode:
            palette_idx = (palette_idx + 1) % len(THERMAL_PALETTES)
            palette_name, _ = THERMAL_PALETTES[palette_idx]
            print(f"Switched to palette: {palette_name}")
        elif key == ord('d'):
            denoise_mode = not denoise_mode
            status = "ON" if denoise_mode else "OFF"
            print(f"Denoise mode: {status}")
        elif key == ord('o'):
            normalize_mode = not normalize_mode
            status = "ON" if normalize_mode else "OFF"
            print(f"Normalize mode: {status}")
        elif key == ord('e'):
            enhance_mode = not enhance_mode
            status = "ON" if enhance_mode else "OFF"
            print(f"Enhance Details (CLAHE) mode: {status}")
        elif key == ord('s'):
            stabilize_mode = not stabilize_mode
            status = "ON" if stabilize_mode else "OFF"
            print(f"Stabilization mode: {status}")
        elif key == ord('S'):
            stabilize_use_orb = not stabilize_use_orb
            method = "ORB+RANSAC" if stabilize_use_orb else "ECC"
            print(f"Stabilization method: {method}")
        elif key == ord('=') and stabilize_mode:
            stabilize_strength = min(1.0, stabilize_strength + 0.1)
            print(f"Stabilization strength: {stabilize_strength:.1f}")
        elif key == ord('-') and stabilize_mode:
            stabilize_strength = max(0.0, stabilize_strength - 0.1)
            print(f"Stabilization strength: {stabilize_strength:.1f}")
        elif key == ord('[') and stabilize_mode:
            stabilize_smooth = max(0.0, stabilize_smooth - 0.05)
            print(f"Stabilization smoothing: {stabilize_smooth:.2f}")
        elif key == ord(']') and stabilize_mode:
            stabilize_smooth = min(0.99, stabilize_smooth + 0.05)
            print(f"Stabilization smoothing: {stabilize_smooth:.2f}")
        elif key == ord('{') and stabilize_mode:
            orb_buffer_size = max(1, orb_buffer_size - 1)
            print(f"Frame buffer size: {orb_buffer_size}")
        elif key == ord('}') and stabilize_mode:
            orb_buffer_size = min(20, orb_buffer_size + 1)
            print(f"Frame buffer size: {orb_buffer_size}")
        elif key == ord('t'):
            threshold_mode = not threshold_mode
            heat_seeker_mode = False
            heat_cluster_mode = False
            motion_mode = False
            palette_mode = False
            yolo_mode = False
            status = "ON" if threshold_mode else "OFF"
            print(f"Threshold mode: {status}")
        elif key == ord('y'):
            if yolo_model is None:
                yolo_model = init_yolo_model()
            if yolo_model:
                yolo_mode = not yolo_mode
                heat_seeker_mode = False
                heat_cluster_mode = False
                motion_mode = False
                palette_mode = False
                threshold_mode = False
                optical_flow_mode = False
                status = "ON" if yolo_mode else "OFF"
                print(f"YOLO AI Detection mode: {status}")
            else:
                print("YOLO model not available")
        elif key == ord('Y'):
            yolo_skip_frames = not yolo_skip_frames
            status = "ON" if yolo_skip_frames else "OFF"
            print(f"YOLO skip-frame optimization: {status}")
        elif key == ord('f'):
            optical_flow_mode = not optical_flow_mode
            heat_seeker_mode = False
            heat_cluster_mode = False
            motion_mode = False
            palette_mode = False
            threshold_mode = False
            yolo_mode = False
            optical_flow_masked_mode = False
            status = "ON" if optical_flow_mode else "OFF"
            print(f"Optical Flow mode: {status}")
        elif key == ord('F'):
            optical_flow_masked_mode = not optical_flow_masked_mode
            heat_seeker_mode = False
            heat_cluster_mode = False
            motion_mode = False
            palette_mode = False
            threshold_mode = False
            yolo_mode = False
            optical_flow_mode = False
            isotherm_mode = False
            status = "ON" if optical_flow_masked_mode else "OFF"
            print(f"Masked Optical Flow mode: {status}")
        elif key == ord('i'):
            isotherm_mode = not isotherm_mode
            heat_seeker_mode = False
            heat_cluster_mode = False
            motion_mode = False
            palette_mode = False
            threshold_mode = False
            yolo_mode = False
            optical_flow_mode = False
            optical_flow_masked_mode = False
            status = "ON" if isotherm_mode else "OFF"
            print(f"Isotherm Highlight mode: {status}")
        elif key == ord('b') and isotherm_mode:
            isotherm_use_black = not isotherm_use_black
            mask_color = "Black" if isotherm_use_black else "Red"
            print(f"Isotherm mask color: {mask_color}")
        elif key == ord('=') and (threshold_mode or optical_flow_masked_mode):
            if optical_flow_masked_mode:
                optical_flow_threshold = min(255, optical_flow_threshold + 5)
                print(f"Optical Flow threshold: {optical_flow_threshold}")
            else:
                threshold_value = min(255, threshold_value + 5)
                print(f"Threshold value: {threshold_value}")
        elif key == ord('-') and (threshold_mode or optical_flow_masked_mode):
            if optical_flow_masked_mode:
                optical_flow_threshold = max(0, optical_flow_threshold - 5)
                print(f"Optical Flow threshold: {optical_flow_threshold}")
            else:
                threshold_value = max(0, threshold_value - 5)
                print(f"Threshold value: {threshold_value}")
        elif isotherm_mode and key != 255:
            # Arrow keys in OpenCV
            if key == 2:  # Left arrow - decrease min
                isotherm_min = max(0, isotherm_min - 5)
                print(f"Isotherm range: {isotherm_min}-{isotherm_max}")
            elif key == 3:  # Right arrow - increase min
                isotherm_min = min(isotherm_max, isotherm_min + 5)
                print(f"Isotherm range: {isotherm_min}-{isotherm_max}")
            elif key == 1:  # Down arrow - decrease max
                isotherm_max = max(isotherm_min, isotherm_max - 5)
                print(f"Isotherm range: {isotherm_min}-{isotherm_max}")
            elif key == 0:  # Up arrow - increase max
                isotherm_max = min(255, isotherm_max + 5)
                print(f"Isotherm range: {isotherm_min}-{isotherm_max}")
        elif heat_seeker_mode and key != 255:
            # Arrow keys for Heat-Seeker box limit and brightness threshold
            if key == 2:  # Left arrow - decrease max boxes
                heat_seeker_max_boxes = max(1, heat_seeker_max_boxes - 1)
                print(f"Heat-Seeker max boxes: {heat_seeker_max_boxes}")
            elif key == 3:  # Right arrow - increase max boxes
                heat_seeker_max_boxes = min(15, heat_seeker_max_boxes + 1)
                print(f"Heat-Seeker max boxes: {heat_seeker_max_boxes}")
            elif key == 1:  # Down arrow - decrease min brightness
                heat_seeker_min_brightness = max(0, heat_seeker_min_brightness - 5)
                print(f"Heat-Seeker min brightness: {heat_seeker_min_brightness}")
            elif key == 0:  # Up arrow - increase min brightness
                heat_seeker_min_brightness = min(255, heat_seeker_min_brightness + 5)
                print(f"Heat-Seeker min brightness: {heat_seeker_min_brightness}")
        
        prev_frame = frame.copy()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

def init_yolo_model():
    """Initialize YOLOv8 Nano model."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("YOLOv8 Nano model loaded successfully")
        return model
    except ImportError:
        print("Error: ultralytics not installed. Run: make install")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def detect_with_yolo(frame, model):
    """Run YOLO detection on frame with Intel optimization."""
    if model is None:
        return frame
    
    try:
        # Use imgsz=320 for Intel speed (~70% faster than default 640)
        results = model.track(frame, persist=True, stream=True, verbose=False, imgsz=320, device='cpu')
        annotated_frame = frame.copy()
        
        for r in results:
            annotated_frame = r.plot()
        
        return annotated_frame
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return frame


if __name__ == "__main__":
    main()
