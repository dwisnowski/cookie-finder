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

def draw_heat_boxes(frame, contours, min_area=100):
    """Draw bounding boxes around hot spots."""
    output = frame.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return output

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
    """Run YOLO detection on frame."""
    if model is None:
        return frame
    
    try:
        results = model.track(frame, persist=True, stream=True, verbose=False)
        annotated_frame = frame.copy()
        
        for r in results:
            annotated_frame = r.plot()
        
        return annotated_frame
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return frame

def main():
    global yolo_model
    
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
    threshold_mode = False
    yolo_mode = False
    threshold_value = 127
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
    print("Press 't' to toggle Threshold mode")
    print("Press 'y' to toggle YOLO AI Detection mode")
    print("Press '=' to increase threshold, '-' to decrease (in Threshold mode)")
    print("Press 'n' to cycle to next palette (in Palette mode)")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        display_frame = frame.copy()
        mode_text = ""
        
        # Apply Normalization if enabled
        if normalize_mode:
            display_frame = normalize_frame(display_frame)
        
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
            display_frame = draw_heat_boxes(display_frame, contours)
            mode_text = "Heat-Seeker: ON"
        
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
        if not heat_seeker_mode and not heat_cluster_mode and not motion_mode and not palette_mode and not threshold_mode and not yolo_mode:
            mode_text = "Normal View"
        
        # Apply YOLO AI Detection if enabled
        if yolo_mode:
            if yolo_model is not None:
                display_frame = detect_with_yolo(display_frame, yolo_model)
                mode_text = "YOLO AI Detection: ON"
        
        # Apply Upscale if enabled (works with all modes)
        if upscale_mode:
            display_frame = upscale_frame(display_frame, scale=2)
        
        # Add denoise indicator
        if denoise_mode:
            mode_text += " | Denoise: ON"
        
        # Add normalize indicator
        if normalize_mode:
            mode_text += " | Normalize: ON"
        
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
                status = "ON" if yolo_mode else "OFF"
                print(f"YOLO AI Detection mode: {status}")
            else:
                print("YOLO model not available")
        elif key == ord('=') and threshold_mode:
            threshold_value = min(255, threshold_value + 5)
            print(f"Threshold value: {threshold_value}")
        elif key == ord('-') and threshold_mode:
            threshold_value = max(0, threshold_value - 5)
            print(f"Threshold value: {threshold_value}")
        
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
    """Run YOLO detection on frame."""
    if model is None:
        return frame
    
    try:
        results = model.track(frame, persist=True, stream=True, verbose=False)
        annotated_frame = frame.copy()
        
        for r in results:
            annotated_frame = r.plot()
        
        return annotated_frame
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return frame
