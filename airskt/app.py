import time
import cv2
import mediapipe as mp
import numpy as np
import os
import warnings
from PIL import Image, ImageDraw, ImageFont  # Import PIL for emojis

# Suppress warnings from dependencies
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress protobuf warnings

from core.tracker import HandTracker
from core.canvas import AirCanvas
from core.utils import OneEuroFilter
from core.gestures import GestureRecognizer
from core.hud import HUDMessage
from core.menu import PaintMenu


# -------------------- UI Helper Functions --------------------
def _draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if thickness == -1:
        # Filled rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    else:
        # Outlined rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y1 + thickness), color, thickness)
        cv2.rectangle(img, (x1 + radius, y2 - thickness), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x1 + thickness, y2 - radius), color, thickness)
        cv2.rectangle(img, (x2 - thickness, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def _draw_startup_instructions(frame, W, H, menu_height):
    """Draw startup gesture instructions for first 5 seconds"""
    overlay = frame.copy()
    
    # Calculate position (center of canvas area) - adjusted for larger box
    inst_x = W // 2 - 220
    inst_y = menu_height + (H - menu_height) // 2 - 180
    
    # Draw semi-transparent background (larger box to fit all instructions)
    bg_overlay = overlay.copy()
    bg_color = (20, 20, 40)
    _draw_rounded_rect(bg_overlay, (inst_x - 20, inst_y - 20), 
                       (inst_x + 440, inst_y + 360), 
                       bg_color, thickness=-1, radius=15)
    cv2.addWeighted(bg_overlay, 0.85, overlay, 0.15, 0, overlay)
    
    # Draw border
    border_color = (100, 200, 255)
    _draw_rounded_rect(overlay, (inst_x - 20, inst_y - 20), 
                       (inst_x + 440, inst_y + 360), 
                       border_color, thickness=3, radius=15)
    
    # Draw title (centered with more top padding)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "GESTURE CONTROLS"
    title_size = cv2.getTextSize(title_text, font, 0.8, 2)[0]
    title_x = inst_x + (400 // 2) - (title_size[0] // 2)  # Center horizontally
    cv2.putText(overlay, title_text, (title_x, inst_y + 25), 
               font, 0.8, (255, 255, 255), 2)
    
    # Draw instructions using PIL for emoji support
    try:
        # Convert to PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a font that supports emojis
        try:
            # Try Windows emoji font
            emoji_font = ImageFont.truetype("seguiemj.ttf", 26)
        except OSError:
            try:
                # Try Mac emoji font
                emoji_font = ImageFont.truetype("Apple Color Emoji.ttc", 26)
            except OSError:
                # Fallback to default
                emoji_font = ImageFont.load_default()
        
        instructions = [
            "☝️Index Up = Draw",
            "✌️Peace Sign = Eraser",
            "✊ Fist = Pause",
            "👋 Swipe Left = Clear",
            "👍 Thumbs Up = Shape",
            "👌 Pinch = Menu",
            "Press H = Help | Q = Quit"
        ]
        
        y_offset = inst_y + 60
        for inst in instructions:
            # Draw text with PIL
            draw.text((inst_x, y_offset), inst, font=emoji_font, fill=(200, 200, 255))
            y_offset += 40
            
        # Convert back to OpenCV format
        overlay = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"⚠️ PIL Drawing failed: {e}")
        # Fallback to OpenCV text
        instructions = [
            "[INDEX UP] = Draw",
            "[PEACE SIGN] = Eraser",
            "[FIST] = Pause",
            "[SWIPE LEFT] = Clear",
            "[THUMBS UP] = Shape",
            "[PINCH] = Menu",
            "Press H = Help | Q = Quit"
        ]
        
        y_offset = inst_y + 60
        for inst in instructions:
            cv2.putText(overlay, inst, (inst_x, y_offset), 
                       font, 0.6, (200, 200, 255), 2)
            y_offset += 30
    
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    return frame

def _draw_status_panel(frame, paused, is_writing, tip_active, W, H):
    """Draw a modern status panel in top-left corner"""
    panel_x, panel_y = 15, 15
    panel_w, panel_h = 280, 120
    
    # Draw semi-transparent background
    overlay = frame.copy()
    bg_color = (25, 25, 40)  # Dark blue-gray
    _draw_rounded_rect(overlay, (panel_x, panel_y), 
                       (panel_x + panel_w, panel_y + panel_h), 
                       bg_color, thickness=-1, radius=12)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw border
    border_color = (100, 150, 255)  # Light blue
    _draw_rounded_rect(frame, (panel_x, panel_y), 
                      (panel_x + panel_w, panel_y + panel_h), 
                      border_color, thickness=2, radius=12)
    
    # Status indicators
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 35
    
    # Mode status
    mode_icon = "⏸️" if paused else "✍️"
    mode_text = f"{mode_icon} {'PAUSED' if paused else 'DRAWING'}"
    mode_color = (100, 150, 255) if paused else (100, 255, 150)  # Blue if paused, green if drawing
    cv2.putText(frame, mode_text, (panel_x + 15, panel_y + y_offset), 
               font, 0.7, mode_color, 2, cv2.LINE_AA)
    
    # Pose status
    y_offset += 30
    pose_icon = "✍️" if is_writing else "✋"
    pose_text = f"{pose_icon} {'Writing Pose' if is_writing else 'Other Gesture'}"
    pose_color = (100, 255, 150) if is_writing else (200, 200, 200)
    cv2.putText(frame, pose_text, (panel_x + 15, panel_y + y_offset), 
               font, 0.55, pose_color, 1, cv2.LINE_AA)
    
    # Tip status
    y_offset += 25
    tip_icon = "●" if tip_active else "○"
    tip_text = f"{tip_icon} Fingertip: {'Active' if tip_active else 'Inactive'}"
    tip_color = (100, 255, 100) if tip_active else (150, 150, 150)
    cv2.putText(frame, tip_text, (panel_x + 15, panel_y + y_offset), 
               font, 0.5, tip_color, 1, cv2.LINE_AA)
    
    # Instructions (bottom right)
    inst_x, inst_y = W - 200, H - 80
    inst_bg = frame.copy()
    _draw_rounded_rect(inst_bg, (inst_x - 10, inst_y - 10), 
                      (inst_x + 190, inst_y + 70), 
                      (20, 20, 30), thickness=-1, radius=10)
    cv2.addWeighted(inst_bg, 0.7, frame, 0.3, 0, frame)
    
    instructions = [
        "✊ Fist = Pause",
        "👋 Swipe = Clear",
        "✌️ Peace = Eraser",
        "Press H = Help"
    ]
    for i, inst in enumerate(instructions):
        cv2.putText(frame, inst, (inst_x, inst_y + i * 20), 
                   font, 0.45, (200, 200, 255), 1, cv2.LINE_AA)

# -------------------- Shape Detection (Improved) --------------------
def detect_shape(image, min_area=100, max_area_ratio=0.8):
    """
    Improved shape detection with:
    - Gaussian blur to reduce noise
    - Morphological operations to clean up artifacts
    - Area, circularity, and convexity filtering
    - Confidence-based classification
    - Multiple thresholding strategies
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if image has any content (not all black)
    max_val = np.max(gray)
    if max_val < 10:
        return None, None
    
    # Apply Gaussian blur to reduce noise (kernel size 5x5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use edge detection first to find drawn lines, then threshold
    # This helps separate drawn lines from background
    edges = cv2.Canny(blurred, 50, 150)
    
    # Try multiple thresholding methods and use the best one
    # Method 1: Otsu's automatic threshold (on original, not edges)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive threshold
    thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    
    # Method 3: Simple threshold (for thin lines) - higher threshold to avoid background
    _, thresh_simple = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    # Method 4: Use edges (only drawn lines, not filled areas)
    thresh_edges = edges.copy()
    
    # Calculate white pixel counts
    otsu_pixels = np.sum(thresh_otsu > 0)
    adapt_pixels = np.sum(thresh_adapt > 0)
    simple_pixels = np.sum(thresh_simple > 0)
    edges_pixels = np.sum(thresh_edges > 0)
    
    # Prefer method with reasonable pixel count (not too many, not too few)
    # Avoid methods that detect entire canvas (too many pixels)
    img_pixels = image.shape[0] * image.shape[1]
    reasonable_max = img_pixels * 0.3  # Max 30% of image should be white
    
    # Prefer edges if it has reasonable pixel count
    if edges_pixels > 100 and edges_pixels < reasonable_max:
        thresh = thresh_edges
        print(f"🔍 Using edge detection: {edges_pixels} pixels")
    elif simple_pixels > 100 and simple_pixels < reasonable_max:
        thresh = thresh_simple
        print(f"🔍 Using simple threshold: {simple_pixels} pixels")
    elif adapt_pixels > 100 and adapt_pixels < reasonable_max:
        thresh = thresh_adapt
        print(f"🔍 Using adaptive threshold: {adapt_pixels} pixels")
    elif otsu_pixels > 100 and otsu_pixels < reasonable_max:
        thresh = thresh_otsu
        print(f"🔍 Using Otsu threshold: {otsu_pixels} pixels")
    else:
        # Fallback: use edges (most likely to work)
        thresh = thresh_edges
        print(f"🔍 Fallback to edges: {edges_pixels} pixels (otsu={otsu_pixels}, adapt={adapt_pixels}, simple={simple_pixels})")
    
    # Morphological operations to clean up noise and close gaps
    kernel = np.ones((3, 3), np.uint8)
    
    # If using edges, we need to close gaps more aggressively
    if np.array_equal(thresh, thresh_edges):
        # For edge detection: close gaps to form closed shapes
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
    else:
        # For threshold methods: standard processing
        # Opening: removes small noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing: fills small gaps (increased iterations for better shape closure)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
        # Dilate slightly to connect nearby lines
        thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    img_area = image.shape[0] * image.shape[1]
    max_allowed_area = max_area_ratio * img_area
    
    # Filter contours by area and get the best one (not necessarily largest)
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_allowed_area:
            valid_contours.append((c, area))
    
    if not valid_contours:
        # If no valid contours, try to find the best one anyway
        print(f"⚠️ No valid contours found, trying largest...")
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > max_allowed_area:
            print(f"⚠️ Largest contour too large: area={area:.0f}, max={max_allowed_area:.0f}, img_area={img_area:.0f}")
            # Try second largest
            if len(contours) > 1:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                for c in sorted_contours[1:]:  # Skip largest, try others
                    area = cv2.contourArea(c)
                    if min_area <= area <= max_allowed_area:
                        print(f"✅ Using second largest contour: area={area:.0f}")
                        break
                else:
                    return None, None
            else:
                return None, None
    else:
        # Use the largest valid contour
        c, area = max(valid_contours, key=lambda x: x[1])
        print(f"✅ Using valid contour: area={area:.0f}")
    
    # Calculate shape properties for confidence
    perimeter = cv2.arcLength(c, True)
    if perimeter < 1:
        return None, None
    
    # Circularity: 4π*area/perimeter² (1.0 = perfect circle)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Convexity: area / convex_hull_area (1.0 = fully convex)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    
    # Approximate contour with adaptive epsilon (very lenient for hand-drawn shapes)
    # Try multiple epsilon values to find the best fit
    # IMPORTANT: Prefer approximations that give 3-4 vertices (triangles/quadrilaterals)
    best_approx = None
    best_vertices = None
    best_priority = -1  # Higher priority = better (3 > 4 > 5-6 > 7+)
    
    # Try epsilon values that are more likely to give 3-4 vertices
    # Start with smaller epsilon (more precise) to catch quadrilaterals
    for eps_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08, 0.10]:  # More granular, starting smaller
        epsilon = eps_factor * perimeter
        approx = cv2.approxPolyDP(c, epsilon, True)
        vertices = len(approx)
        
        # Priority: 3 vertices (triangle) > 4 vertices (square/rect) > 5-6 > 7+
        if vertices == 3:
            priority = 10  # Highest priority
        elif vertices == 4:
            priority = 9   # Second highest (squares/rectangles) - CRITICAL
        elif vertices == 5 or vertices == 6:
            priority = 5   # Medium (could be circle or polygon)
        elif 3 <= vertices <= 8:
            priority = 3   # Lower priority
        else:
            priority = 1   # Lowest priority
        
        if priority > best_priority:
            best_approx = approx
            best_vertices = vertices
            best_priority = priority
            # If we found a perfect match (3 or 4 vertices), use it immediately
            if vertices == 3 or vertices == 4:
                print(f"🔍 Found {vertices} vertices with epsilon={eps_factor:.3f} - using this")
                break
    
    approx = best_approx
    vertices = best_vertices if best_vertices else len(approx)
    
    # Debug info
    print(f"🔍 Shape detection: area={area:.0f}, perimeter={perimeter:.0f}, vertices={vertices}, "
          f"circularity={circularity:.3f}, convexity={convexity:.3f}")
    
    # Classify shape with confidence checks (more lenient thresholds)
    shape = None
    confidence = "Low"
    
    # Classify shape - STRICT priority: vertices count FIRST, then properties
    # This prevents misclassification (e.g., squares as circles)
    
    # STEP 1: Check by vertex count FIRST (most reliable)
    if vertices == 3:
        # Triangle: 3 vertices = triangle (almost always)
        if convexity > 0.50:  # Very lenient
            shape = "Triangle"
            confidence = "High" if convexity > 0.80 else "Medium" if convexity > 0.65 else "Low"
            print(f"✅ Triangle detected: convexity={convexity:.3f}")
    elif vertices == 4:
        # Quadrilateral: 4 vertices = square or rectangle (ALWAYS, regardless of circularity)
        # This is CRITICAL - 4 vertices means it's a quadrilateral, not a circle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h > 0 else 1.0
        
        # Classify as square or rectangle based on aspect ratio
        # Very lenient thresholds to catch all quadrilaterals
        if 0.60 <= aspect_ratio <= 1.60:  # Very lenient for squares
            shape = "Square" if 0.75 <= aspect_ratio <= 1.25 else "Rectangle"
            confidence = "High" if 0.85 <= aspect_ratio <= 1.15 else "Medium" if 0.70 <= aspect_ratio <= 1.40 else "Low"
            print(f"✅ Quadrilateral detected: {shape}, aspect_ratio={aspect_ratio:.3f}, convexity={convexity:.3f}, circularity={circularity:.3f} (4 vertices = always quadrilateral)")
        else:
            shape = "Rectangle"
            confidence = "Medium" if convexity > 0.50 else "Low"
            print(f"✅ Rectangle detected: aspect_ratio={aspect_ratio:.3f}, convexity={convexity:.3f} (4 vertices = always quadrilateral)")
    elif vertices == 5 or vertices == 6:
        # 5-6 vertices: Could be pentagon/hexagon, circle, OR imperfect square/rectangle
        # Check aspect ratio FIRST - if it's close to 1:1 or rectangular, it's likely a square/rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h > 0 else 1.0
        
        # If aspect ratio suggests square/rectangle (even with extra vertices), classify as such
        if 0.60 <= aspect_ratio <= 1.60 and convexity > 0.60:
            # Likely an imperfect square/rectangle (hand-drawn with extra corners)
            shape = "Square" if 0.75 <= aspect_ratio <= 1.25 else "Rectangle"
            confidence = "Medium"  # Medium confidence due to extra vertices
            print(f"✅ Quadrilateral detected (imperfect, {vertices} vertices): {shape}, aspect_ratio={aspect_ratio:.3f}, convexity={convexity:.3f}")
        elif circularity > 0.50:  # Must be fairly circular
            shape = "Circle"
            confidence = "High" if circularity > 0.70 else "Medium" if circularity > 0.60 else "Low"
            print(f"✅ Circle detected: circularity={circularity:.3f}, vertices={vertices}")
        else:
            # Not circular enough and not rectangular - might be a polygon, but classify as circle with low confidence
            shape = "Circle"
            confidence = "Low"
            print(f"✅ Circle detected (low confidence): circularity={circularity:.3f}, vertices={vertices}")
    elif vertices >= 7:
        # 7+ vertices: Likely a circle (or complex polygon)
        if circularity > 0.40:  # Very lenient
            shape = "Circle"
            confidence = "High" if circularity > 0.70 else "Medium" if circularity > 0.55 else "Low"
            print(f"✅ Circle detected: circularity={circularity:.3f}, vertices={vertices}")
    
    # If no shape matched, try alternative detection methods
    if shape is None:
        print(f"⚠️ Primary detection failed: vertices={vertices}, circularity={circularity:.3f}, convexity={convexity:.3f}")
        
        # Alternative 1: Try with even more lenient thresholds
        if vertices == 4 and convexity > 0.50:
            # Very lenient rectangle/square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 1.0
            if 0.70 <= aspect_ratio <= 1.40:
                shape = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
                confidence = "Low"
                print(f"✅ Alternative detection: {shape} (low confidence)")
                return shape, c
            else:
                shape = "Rectangle"
                confidence = "Low"
                print(f"✅ Alternative detection: {shape} (low confidence)")
                return shape, c
        
        # Alternative 2: Circle with very lenient circularity
        if vertices >= 4 and circularity > 0.40:
            shape = "Circle"
            confidence = "Low"
            print(f"✅ Alternative detection: {shape} (low confidence)")
            return shape, c
        
        # Alternative 3: Triangle with lenient convexity
        if vertices == 3 and convexity > 0.60:
            shape = "Triangle"
            confidence = "Low"
            print(f"✅ Alternative detection: {shape} (low confidence)")
            return shape, c
        
        # Last resort: Return Unknown
        print(f"⚠️ No shape matched after all attempts")
        return "Unknown", c
    
    # Return shape even with low confidence (user can see what was detected)
    print(f"✅ Detected: {shape} (confidence: {confidence})")
    return shape, c


def detect_shape_enhanced(image, min_area=80, max_area_ratio=0.8):
    """
    Enhanced shape detection with more aggressive preprocessing for difficult cases.
    Uses dilation and erosion to close gaps in hand-drawn shapes.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if image has any content
    max_val = np.max(gray)
    if max_val < 10:
        return None, None
    
    # More aggressive blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Try multiple thresholding methods
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 15, 3)
    _, thresh_simple = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    
    # Use the best threshold
    otsu_pixels = np.sum(thresh_otsu > 0)
    adapt_pixels = np.sum(thresh_adapt > 0)
    simple_pixels = np.sum(thresh_simple > 0)
    
    if otsu_pixels >= adapt_pixels and otsu_pixels >= simple_pixels:
        thresh = thresh_otsu
    elif adapt_pixels >= simple_pixels:
        thresh = thresh_adapt
    else:
        thresh = thresh_simple
    
    # Very aggressive morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    # Dilate to thicken lines
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    # Close gaps aggressively
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    # Erode back slightly
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    # Get the largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    img_area = image.shape[0] * image.shape[1]
    
    if area < min_area or area > (max_area_ratio * img_area):
        return None, None
    
    # Calculate properties
    perimeter = cv2.arcLength(c, True)
    if perimeter < 1:
        return None, None
    
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    
    # Very lenient approximation
    epsilon = 0.10 * perimeter  # 10% - very lenient
    approx = cv2.approxPolyDP(c, epsilon, True)
    vertices = len(approx)
    
    print(f"🔍 Enhanced detection: area={area:.0f}, vertices={vertices}, circularity={circularity:.3f}, convexity={convexity:.3f}")
    
    # Very lenient classification
    shape = None
    if vertices == 3:
        if convexity > 0.55:
            shape = "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h > 0 else 1.0
        if convexity > 0.55:
            if 0.70 <= aspect_ratio <= 1.40:
                shape = "Square" if 0.80 <= aspect_ratio <= 1.20 else "Rectangle"
            else:
                shape = "Rectangle"
        else:
            shape = "Rectangle"
    elif vertices >= 5:  # 5+ vertices for circles
        if circularity > 0.35:  # Very lenient
            shape = "Circle"
    
    if shape:
        print(f"✅ Enhanced detection succeeded: {shape}")
        return shape, c
    
    return None, None


# -------------------- Main Application --------------------
def main():
    # --- Camera setup (larger resolution for bigger canvas) ---
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return
    print("✅ Camera opened successfully (1280x720)!")

    tracker = HandTracker()
    # Get actual frame dimensions after camera setup
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Could not read test frame.")
        return
    W, H = test_frame.shape[1], test_frame.shape[0]  # Use actual frame dimensions
    w, h = W, H  # Keep for compatibility
    
    # Create menu and adjust canvas size
    menu = PaintMenu(menu_height=120, canvas_width=W)  # Use actual frame width
    canvas_height = H - menu.menu_height  # Leave space for top menu
    canvas = AirCanvas(width=W, height=canvas_height)  # Match frame dimensions exactly
    
    # Initialize canvas with menu defaults
    canvas.set_color(menu.get_selected_color())
    canvas.set_thickness(menu.get_selected_brush_size())

    # OPTIMIZED FILTER: Increased beta (0.002 -> 0.05) for much faster response
    # Increased min_cutoff (1.0 -> 1.5) to keep it stable when moving slowly
    fx = OneEuroFilter(min_cutoff=1.5, beta=0.05)
    fy = OneEuroFilter(min_cutoff=1.5, beta=0.05)
    
    gestures = GestureRecognizer()
    hud = HUDMessage()
    
    # Startup instructions
    startup_time = time.time()
    show_startup = True
    manual_help_toggle = False  # Track if user manually toggled help
    
    # Menu interaction state
    last_pinch_time = 0
    pinch_cooldown = 0.5  # Prevent rapid toggling (500ms)
    eraser_active = False
    last_hand_y = None  # For eraser size scrolling

    paused = False
    last_clear_time = 0
    clear_cooldown = 0.3  # Reduced from 0.5s for faster recovery after clear
    frame_count = 0
    gesture_cooldown = 0.3  # Fast gesture response
    last_gesture_time = 0
    last_tip = None
    last_tip_t = 0
    pause_time = None  # track when pause started
    detected_shape_contour = None  # Store contour for visualization
    shape_detection_time = None  # Track when shape was detected
    max_pause_time = 3.0  # Maximum time to stay paused (prevent getting stuck)
    last_shape_detection_time = 0  # Track last time we ran shape detection
    shape_detection_cooldown = 2.0  # Cooldown between shape detections (2 seconds)
    is_writing_pose = False  # Track if hand is in writing pose
    last_shape_draw_time = 0  # Track when last shape was drawn
    shape_draw_cooldown = 1.0  # Cooldown between shape draws (1 second gap)
    resizing_shape = False  # Track if currently resizing a shape
    resize_start_y = None  # Track starting Y position for resize
    currently_analyzing_shape = False  # Track if ACTIVELY analyzing shape NOW
    last_shape_size = None  # Track original size when resize started
    resize_start_time = None  # Track when resize started (for timeout)
    was_pinching = False  # Track previous pinch state to detect release
    pending_shape_draw = False  # Track if shape should be drawn on pinch release

    # Comfort Mode (Anti-Fatigue) Configuration
    comfort_mode = True
    active_ratio_w = 0.7  # Use 70% of width
    active_ratio_h = 0.6  # Use 60% of height
    
    def get_mapped_coords(point_normalized, W, H):
        """Map normalized coordinates from a smaller active area to the full screen to reduce arm fatigue."""
        if not comfort_mode:
            return int(point_normalized.x * W), int(point_normalized.y * H)
            
        # Center the active area, slightly shifted down (hands are usually lower)
        start_x_ratio = (1.0 - active_ratio_w) / 2
        start_y_ratio = (1.0 - active_ratio_h) / 2 + 0.05
        
        # Normalize to 0-1 within the active area
        norm_x = (point_normalized.x - start_x_ratio) / active_ratio_w
        norm_y = (point_normalized.y - start_y_ratio) / active_ratio_h
        
        # Clamp to [0, 1] to keep cursor within screen
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        return int(norm_x * W), int(norm_y * H)

    print("✅ AirSketch – Stable v3 (Comfort Mode Active)")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_count += 1
        
        # Draw Comfort Mode Active Area box
        if comfort_mode:
            start_x_ratio = (1.0 - active_ratio_w) / 2
            start_y_ratio = (1.0 - active_ratio_h) / 2 + 0.05
            
            # Calculate pixel coordinates for the box
            box_x1 = int(start_x_ratio * W)
            box_y1 = int(start_y_ratio * H)
            box_w = int(active_ratio_w * W)
            box_h = int(active_ratio_h * H)
            
            # Draw semi-transparent box or corners
            # Draw corners to be less intrusive
            corner_len = 30
            color = (100, 255, 100) # Light Green
            thickness = 2
            
            # Top-Left
            cv2.line(frame, (box_x1, box_y1), (box_x1 + corner_len, box_y1), color, thickness)
            cv2.line(frame, (box_x1, box_y1), (box_x1, box_y1 + corner_len), color, thickness)
            # Top-Right
            cv2.line(frame, (box_x1 + box_w, box_y1), (box_x1 + box_w - corner_len, box_y1), color, thickness)
            cv2.line(frame, (box_x1 + box_w, box_y1), (box_x1 + box_w, box_y1 + corner_len), color, thickness)
            # Bottom-Left
            cv2.line(frame, (box_x1, box_y1 + box_h), (box_x1 + corner_len, box_y1 + box_h), color, thickness)
            cv2.line(frame, (box_x1, box_y1 + box_h), (box_x1, box_y1 + box_h - corner_len), color, thickness)
            # Bottom-Right
            cv2.line(frame, (box_x1 + box_w, box_y1 + box_h), (box_x1 + box_w - corner_len, box_y1 + box_h), color, thickness)
            cv2.line(frame, (box_x1 + box_w, box_y1 + box_h), (box_x1 + box_w, box_y1 + box_h - corner_len), color, thickness)
            
            cv2.putText(frame, "Active Area", (box_x1 + 5, box_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        gesture, pos = None, None
        hand_detected = False
        now = time.time()
        
        # Process hand tracking EVERY FRAME (not every other frame) for better responsiveness
        small = cv2.resize(frame, (640, 360))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = tracker.hands.process(rgb_small)

        # Initialize variables
        is_pinch = False
        is_two_finger = False
        is_writing_pose = False
        hand_landmarks = None
        
        # ALWAYS detect gestures (even when paused, to allow resume with fist)
        g, p = gestures.detect_gesture(results, (H, W, 3))
        if g and (now - last_gesture_time) > gesture_cooldown:
            gesture, pos = g, p
            last_gesture_time = now
        
        # Extract gesture states IMMEDIATELY (before any other logic)
        if p and isinstance(p, dict):
            is_writing_pose = p.get("is_writing", False)
            is_pinch = p.get("is_pinch", False)
            is_two_finger = p.get("is_two_finger", False)
            hand_landmarks = p.get("hand_landmarks", None)

        # CRITICAL: If paused, skip ALL hand tracking, menu interaction, and drawing
        if paused:
            # Completely stop all tracking when paused - reset everything EVERY FRAME
            canvas.prev_point = None
            last_tip = None
            last_tip_t = 0
            # Skip all hand processing below - no menu interaction, no drawing, no tracking
        else:
            # Only process hand info when NOT paused (states already extracted above)
            if results.multi_hand_landmarks:
                # If hand detected but no pose info, check directly
                hand = results.multi_hand_landmarks[0]
                # More lenient check: index up is primary requirement
                # Allow drawing if index is up, even if other fingers aren't perfectly down
                lm = hand.landmark
                index_up = lm[8].y < lm[6].y
                # At least 2 other fingers should be down (more lenient than requiring all 3)
                fingers_down_count = sum([
                    lm[12].y > lm[10].y,  # middle down
                    lm[16].y > lm[14].y,  # ring down
                    lm[20].y > lm[18].y   # pinky down
                ])
                is_writing_pose = index_up and fingers_down_count >= 2
            
            # Handle menu interaction with pinch gesture (only when not paused)
            if is_pinch and hand_landmarks and (now - last_pinch_time) > pinch_cooldown:
                # Get index finger tip position (using mapped coordinates)
                index_tip = hand_landmarks[8]
                tip_x, tip_y = get_mapped_coords(index_tip, W, H)
                
                # Check if pinching on menu area (top bar)
                if tip_y < menu.menu_height:
                    # Check color selection
                    color_idx = menu.check_color_selection(tip_x, tip_y)
                    if color_idx is not None:
                        menu.selected_color_idx = color_idx
                        selected_color = menu.get_selected_color()
                        canvas.set_color(selected_color)
                        eraser_active = False  # Reset eraser when selecting color
                        menu.selected_tool_idx = 0  # Switch to Draw mode
                        hud.show(f"[COLOR] {menu.colors[color_idx][0]}")
                        last_pinch_time = now
                        print(f"✅ Selected color: {menu.colors[color_idx][0]} -> {selected_color}")
                    
                    # Check brush selection
                    brush_idx = menu.check_brush_selection(tip_x, tip_y)
                    if brush_idx is not None:
                        menu.selected_brush_idx = brush_idx
                        brush_size = menu.get_selected_brush_size()
                        canvas.set_thickness(brush_size)
                        hud.show(f"[BRUSH] {brush_size}px")
                        last_pinch_time = now
                        print(f"✅ Selected brush: {brush_size}px")
                    
                    # Check tool selection
                    tool_idx = menu.check_tool_selection(tip_x, tip_y)
                    if tool_idx is not None:
                        menu.selected_tool_idx = tool_idx
                        if tool_idx == 1:  # Eraser
                            canvas.eraser_mode = True  # Set eraser mode directly
                            eraser_active = True
                            hud.show(f"[ERASER] Mode ({menu.get_selected_brush_size()}px)")
                        else:
                            canvas.eraser_mode = False  # Disable eraser mode
                            canvas.set_color(menu.get_selected_color())
                            eraser_active = False
                            hud.show("[DRAW] Mode")
                        last_pinch_time = now
                    
                    # Check shape selection
                    shape_idx = menu.check_shape_selection(tip_x, tip_y)
                    if shape_idx is not None:
                        # Always select the clicked shape (no toggle)
                        menu.selected_shape_idx = shape_idx
                        hud.show(f"[SHAPE] {menu.shapes[shape_idx]} selected")
                        print(f"✅ Shape mode: {menu.shapes[shape_idx]} - selected_shape_idx={menu.selected_shape_idx}")
                        last_pinch_time = now
                        # Don't draw shape when selecting from menu
                        pending_shape_draw = False
                    
                    # Check Draw tool selection (to deselect shapes)
                    tool_idx = menu.check_tool_selection(tip_x, tip_y)
                    if tool_idx is not None and tool_idx == 0:  # Draw tool
                        # Deselect shape when Draw tool is selected
                        menu.selected_shape_idx = None
                        hud.show("[TOOL] Freehand mode")
                        print("✅ Freehand mode - shape deselected")
            
            # Handle two-finger eraser (only when not paused) - Peace Sign ✌️
            if is_two_finger:
                eraser_active = True
                # Get hand landmarks directly from results
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    # Get index finger tip for erasing (using mapped coordinates)
                    index_tip = hand.landmark[8]
                    tip_x, tip_y = get_mapped_coords(index_tip, W, H)
                    # Only erase in canvas area (not menu - check Y coordinate)
                    if tip_y > menu.menu_height:
                        # Use brush size for eraser
                        eraser_radius = menu.get_selected_brush_size() * 3
                        canvas.erase_point((tip_x, tip_y - menu.menu_height), radius=eraser_radius)
                        # Show visual feedback
                        cv2.circle(frame, (tip_x, tip_y), eraser_radius, (255, 100, 100), 2)
                        # Debug feedback
                        if frame_count % 10 == 0:  # Print every 10 frames
                            print(f"✌️ Eraser active at ({tip_x}, {tip_y}) with radius {eraser_radius}")
            
            # Update fingertip if detected (for both drawing and shape placement) - separate from eraser
            if results.multi_hand_landmarks and not is_two_finger:
                hand_detected = True
                hand = results.multi_hand_landmarks[0]
                tip = hand.landmark[8]
                tip_x, tip_y = get_mapped_coords(tip, W, H)
                
                # Always update last_tip for shape placement (even without writing pose)
                # Only draw in canvas area (not menu - check Y coordinate)
                if tip_y > menu.menu_height:
                    # Update fingertip position (needed for shape drawing)
                    last_tip = (tip_x, tip_y - menu.menu_height)  # Adjust for menu offset
                    last_tip_t = now
                    
                    # For freehand drawing, still require writing pose
                    if is_writing_pose:
                        # In writing pose - ready for freehand drawing
                        pass
                    else:
                        # Not in writing pose - reset drawing point for freehand
                        canvas.prev_point = None
                else:
                    # Update time even if outside canvas area
                    last_tip_t = now
            elif not results.multi_hand_landmarks:
                # Hand not detected - reset writing pose status
                is_writing_pose = False
                # Reset fingertip if hand lost during drawing (prevents stale drawing)
                canvas.prev_point = None
                last_tip = None
                last_tip_t = 0

        # ---- Handle gestures ----
        if gesture == "fist":
            # Toggle pause: fist gesture pauses if not paused, resumes if paused
            if not paused:
                paused = True
                pause_time = time.time()
                # CRITICAL: Immediately reset all drawing state
                canvas.prev_point = None
                last_tip = None
                last_tip_t = 0
                hud.show("[PAUSE] Hold steady...")
                print(f"✋ Pause triggered - last_tip={last_tip}, prev_point={canvas.prev_point}")
            else:
                # Resume from pause
                paused = False
                pause_time = None
                canvas.prev_point = None
                last_tip = None
                last_tip_t = 0
                hud.show("[RESUME] Drawing resumed")
                print("✋ Resume triggered")

        elif gesture == "sweep_left":
            canvas.clear()
            hud.show("[CLEAR] Canvas Cleared")
            last_clear_time = time.time()
            canvas.prev_point = None
            last_tip = None  # Reset fingertip tracking after clear
            last_tip_t = 0
            # Also reset pause state if stuck
            if paused:
                paused = False
                pause_time = None

        elif gesture == "thumbs_up":
            # Check cooldown to prevent running detection every frame
            now = time.time()
            if now - last_shape_detection_time >= shape_detection_cooldown:
                # Mark that we're ACTIVELY analyzing shape NOW
                currently_analyzing_shape = True
                last_shape_detection_time = now
                
                # Trigger shape recognition
                hud.show("[SHAPE] Analyzing...")
                print(f"👍 Analyzing shape... (cooldown passed, last={last_shape_detection_time})")
                
                # Run detection
                shape, contour = detect_shape(canvas.image)
            
                if shape and shape != "Unknown":
                    detected_shape_contour = contour
                    shape_detection_time = time.time()
                    hud.show(f"[SHAPE] Detected: {shape}")
                    print(f"✅ Detected: {shape}")
                else:
                    # Try enhanced detection if simple failed
                    print(f"⚠️ First attempt failed, trying enhanced detection...")
                    shape2, contour2 = detect_shape_enhanced(canvas.image)
                    if shape2 and shape2 != "Unknown":
                        detected_shape_contour = contour2
                        shape_detection_time = time.time()
                        hud.show(f"[SHAPE] Detected: {shape2}")
                        print(f"✅ Enhanced detection succeeded: {shape2}")
                    else:
                        hud.show(f"[SHAPE] Unknown shape")
                        print(f"⚠️ Could not detect shape")
            else:
                # Still in cooldown
                currently_analyzing_shape = False
        else:
            # Not showing thumbs up anymore, so NOT analyzing
            currently_analyzing_shape = False

        # ---- Drawing behavior (only when NOT paused) ----
        if not paused:
            # Check if shape mode is active
            selected_shape = menu.get_selected_shape()
            
            # Handle shape drawing
            if selected_shape and last_tip is not None:
                now = time.time()
                
                # Detect pinch release (was pinching, now not pinching)
                pinch_released = was_pinching and not is_pinch
                
                # Get current position (already adjusted for menu offset in last_tip)
                x = int(fx.filter(last_tip[0]))
                y = int(fy.filter(last_tip[1]))  # This is already canvas-relative (y - menu.menu_height)
                
                # Convert to screen coordinates for menu check
                screen_y = y + menu.menu_height
                
                # Debug: track resizing_shape state
                if is_pinch and len(canvas.shapes) > 0:
                    print(f"🔍 Frame check: resizing_shape={resizing_shape}, is_pinch={is_pinch}, screen_y={screen_y}, time_since_draw={now - last_shape_draw_time:.2f}")
                
                # PRIORITY 1: Handle resize mode (check this FIRST before other pinch actions)
                if resizing_shape:
                    # Check for timeout (2 seconds)
                    if resize_start_time and (now - resize_start_time) >= 2.0:
                        # Auto-stop after 2 seconds
                        print(f"🔍 Resize STOP: 2 second timeout")
                        resizing_shape = False
                        resize_start_y = None
                        last_shape_size = None
                        resize_start_time = None
                        hud.show("[RESIZE] Done")
                    # We're already resizing - continue or stop
                    elif is_pinch and len(canvas.shapes) > 0 and screen_y > menu.menu_height:
                        # Continue resizing based on vertical movement
                        if resize_start_y is not None and last_shape_size is not None:
                            dy = resize_start_y - screen_y  # Positive = moved up (larger), negative = moved down (smaller)
                            size_delta = int(dy * 2.0)  # More sensitive: 2x ratio for better responsiveness
                            new_size = last_shape_size + size_delta
                            new_size = max(10, min(new_size, 500))  # Clamp between 10 and 500
                            
                            # Get current size to compare
                            current_shape = canvas.get_last_shape()
                            current_size = current_shape['size'] if current_shape else last_shape_size
                            
                            # Resize if size changed
                            if abs(new_size - current_size) >= 1:
                                result = canvas.resize_last_shape(new_size)
                                if result:
                                    hud.show(f"[RESIZE] Size: {new_size}")
                                    print(f"🔍 Resize UPDATE: dy={dy:.1f}, delta={size_delta}, new_size={new_size}, current_size={current_size}, start_size={last_shape_size}, resizing_shape={resizing_shape}")
                                else:
                                    print(f"⚠️ Resize failed: result={result}")
                        else:
                            print(f"⚠️ Resize state invalid: resize_start_y={resize_start_y}, last_shape_size={last_shape_size}")
                    # If not pinching but resizing_shape is True, keep it True (don't reset)
                # PRIORITY 2: Start resize mode (if not already resizing)
                elif is_pinch and len(canvas.shapes) > 0 and screen_y > menu.menu_height and (now - last_shape_draw_time) < 5.0 and not resizing_shape:
                    # Start resizing - capture initial state
                    resizing_shape = True
                    resize_start_y = screen_y  # Use screen coordinates for tracking
                    resize_start_time = now  # Track when resize started
                    last_shape = canvas.get_last_shape()
                    if last_shape:
                        last_shape_size = last_shape['size']
                        hud.show("[RESIZE] Move hand up/down")
                        print(f"🔍 Resize START: shape_size={last_shape_size}, start_y={resize_start_y}, shapes_count={len(canvas.shapes)}, resizing_shape={resizing_shape}")
                    else:
                        print(f"⚠️ No shape found to resize")
                        resizing_shape = False
                        resize_start_time = None
                # PRIORITY 3: Stop resizing when pinch is released
                elif resizing_shape and not is_pinch:
                    # Stop resizing when pinch is released
                    print(f"🔍 Resize STOP: pinch released, was resizing")
                    resizing_shape = False
                    resize_start_y = None
                    last_shape_size = None
                    resize_start_time = None
                    hud.show("[RESIZE] Done")
                elif pinch_released and screen_y > menu.menu_height and (now - last_shape_draw_time) > shape_draw_cooldown:
                    # Draw shape when pinch is released on canvas (only once, no writing pose required)
                    shape_size = menu.get_selected_brush_size() * 8  # Scale shape size with brush
                    canvas.draw_shape(selected_shape, (x, y), shape_size)
                    last_shape_draw_time = now
                    canvas.prev_point = None  # Reset for next shape
                    hud.show(f"[SHAPE] {selected_shape} drawn")
                    print(f"✅ Shape drawn: {selected_shape} at ({x}, {y}) size={shape_size}")
                    # Keep shape selected so user can draw multiple shapes
                    # menu.selected_shape_idx = None
                    # Reset resize state when drawing new shape
                    resizing_shape = False
                    resize_start_y = None
                    last_shape_size = None
                    resize_start_time = None
                elif is_pinch and screen_y > menu.menu_height:
                    # Pinch detected on canvas - show feedback
                    hud.show(f"[PINCH] Release to draw {selected_shape}")
            elif last_tip is not None and not selected_shape and not paused:
                # Freehand drawing mode (ONLY when not paused)
                if (time.time() - last_clear_time > clear_cooldown and 
                    time.time() - last_tip_t < 0.3):  # Increased timeout slightly for smoother drawing
                    x = int(fx.filter(last_tip[0]))
                    y = int(fy.filter(last_tip[1]))
                    canvas.draw_point((x, y))
                elif time.time() - last_tip_t >= 0.3:
                    # Reset if tip is stale or not available
                    if canvas.prev_point is not None:
                        canvas.prev_point = None
            elif last_tip is None or time.time() - last_tip_t >= 0.3:
                # Reset if tip is stale or not available
                if canvas.prev_point is not None:
                    canvas.prev_point = None
                
                # NOTE: Shape detection removed from normal drawing
                # Shape detection now ONLY happens when paused (see pause handling below)
        # Handle pause state (separate from drawing logic)
        if paused:
            # stay paused briefly
            if pause_time:
                elapsed = time.time() - pause_time
                # Force resume if stuck too long
                if elapsed >= max_pause_time:
                    print("⚠️ Pause timeout - resuming")
                    paused = False
                    pause_time = None
                    canvas.prev_point = None
                    hud.show("[RESUME] Auto-resumed")
                else:
                    # Just show paused status
                    pass
            
        # Update pinch state for next frame (after if/else block)
        was_pinching = is_pinch

        # ---- Compose Output with better blending ----
        # Canvas should match frame dimensions exactly (no resizing needed = no blurriness)
        canvas_resized = canvas.image  # Direct use, no resizing
        # Blend canvas with camera feed (only in canvas area, below menu)
        output = frame.copy()
        canvas_area = output[menu.menu_height:, :]
        canvas_area = cv2.addWeighted(canvas_area, 0.75, canvas_resized, 0.25, 0)
        output[menu.menu_height:, :] = canvas_area
        
        # Draw menu at top
        output = menu.draw(output)
        
        # Draw cursor/hand indicator on menu for visual feedback
        if results.multi_hand_landmarks and not paused:
            hand = results.multi_hand_landmarks[0]
            index_tip = hand.landmark[8]
            cursor_x, cursor_y = get_mapped_coords(index_tip, W, H)
            
            # Draw cursor circle on screen (always visible)
            cv2.circle(output, (cursor_x, cursor_y), 12, (0, 255, 255), 2)
            cv2.circle(output, (cursor_x, cursor_y), 8, (0, 255, 255), -1)
            
            # If cursor is in menu area, show what would be selected
            if cursor_y < menu.menu_height:
                # Check which menu item is being hovered
                color_idx = menu.check_color_selection(cursor_x, cursor_y)
                brush_idx = menu.check_brush_selection(cursor_x, cursor_y)
                tool_idx = menu.check_tool_selection(cursor_x, cursor_y)
                shape_idx = menu.check_shape_selection(cursor_x, cursor_y)
                
                # Draw hover highlight
                if color_idx is not None:
                    # Highlight color box
                    start_x = 20
                    box_size = 45
                    spacing = 10
                    start_y = 35
                    box_x = start_x + color_idx * (box_size + spacing)
                    cv2.rectangle(output, (box_x - 3, start_y - 3), 
                                (box_x + box_size + 3, start_y + box_size + 3),
                                (255, 255, 0), 3)  # Yellow highlight
                elif brush_idx is not None:
                    # Highlight brush circle
                    brush_start_x = 20 + len(menu.colors) * (45 + 10) + 25
                    center_x = brush_start_x + brush_idx * (18 + menu.brush_spacing) + 12
                    center_y = 35 + 45 // 2
                    cv2.circle(output, (center_x, center_y), 18, (255, 255, 0), 3)
                elif tool_idx is not None:
                    # Highlight tool box
                    colors_end = 20 + len(menu.colors) * (45 + 10)
                    brushes_end = colors_end + 25 + len(menu.brush_sizes) * (18 + menu.brush_spacing) + 20
                    start_x = brushes_end + 10
                    start_y = 35
                    box_x = start_x + tool_idx * (70 + 8)
                    cv2.rectangle(output, (box_x - 3, start_y - 3),
                                (box_x + 70 + 3, start_y + 38 + 3),
                                (255, 255, 0), 3)  # Yellow highlight
                elif shape_idx is not None:
                    # Highlight shape box
                    colors_end = 20 + len(menu.colors) * (45 + 10)
                    brushes_end = colors_end + 25 + len(menu.brush_sizes) * (18 + menu.brush_spacing) + 20
                    tools_end = brushes_end + 10 + len(menu.tools) * (70 + 8)
                    start_x = tools_end + 10
                    start_y = 35
                    box_x = start_x + shape_idx * (65 + 6)
                    cv2.rectangle(output, (box_x - 3, start_y - 3),
                                (box_x + 65 + 3, start_y + 38 + 3),
                                (255, 255, 0), 3)  # Yellow highlight

        # Show BIG visual indicator when ACTIVELY analyzing shape (thumbs up held)
        if currently_analyzing_shape:
            # Big yellow "ANALYZING..." text in center
            text = "ANALYZING SHAPE..."
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.5
            thickness = 4
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (W - text_size[0]) // 2
            text_y = H // 2
            # Draw black background for contrast
            cv2.rectangle(output, (text_x - 20, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 20, text_y + 10),
                         (0, 0, 0), -1)
            # Draw yellow text
            cv2.putText(output, text, (text_x, text_y),
                       font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        
        # Draw detected shape contour if available (visual feedback with glow effect)
        # ONLY show if shape was detected via thumbs up gesture
        if detected_shape_contour is not None and shape_detection_time is not None:
            # Scale contour to match output size
            scale_x = W / canvas.width
            scale_y = H / canvas.height
            scaled_contour = detected_shape_contour.copy()
            scaled_contour[:, :, 0] = (scaled_contour[:, :, 0] * scale_x).astype(np.int32)
            scaled_contour[:, :, 1] = (scaled_contour[:, :, 1] * scale_y).astype(np.int32)
            
            # Draw with glow effect (thicker outer line + inner line)
            glow_color = (0, 200, 255)  # Bright cyan
            main_color = (0, 255, 255)  # Cyan
            cv2.drawContours(output, [scaled_contour], -1, glow_color, 5)  # Outer glow
            cv2.drawContours(output, [scaled_contour], -1, main_color, 2)  # Inner line
            
            # Clear after 3 seconds (increased visibility)
            if time.time() - shape_detection_time > 3.0:
                detected_shape_contour = None
                shape_detection_time = None

        # Draw startup instructions (first 5 seconds, or indefinitely if manually toggled)
        if show_startup:
            if manual_help_toggle or (time.time() - startup_time) < 5.0:
                output = _draw_startup_instructions(output, W, H, menu.menu_height)
            else:
                # Auto-hide after 5 seconds (only if not manually toggled)
                show_startup = False
        
        # Draw HUD messages in bottom right corner
        output = hud.draw(output)

        cv2.imshow("AirSketch – Stable v3", output)
        key = cv2.waitKey(1) & 0xFF
        
        # Keyboard controls
        if key == ord("q"):
            break
        elif key == ord("h") or key == ord("H"):
            # Toggle help/instructions display
            show_startup = not show_startup
            if show_startup:
                manual_help_toggle = True  # User manually toggled, don't auto-hide
                startup_time = time.time()  # Reset timer
                hud.show("[HELP] Showing instructions")
                print("ℹ️ Help menu toggled ON")
            else:
                manual_help_toggle = False  # Reset flag when hiding
                hud.show("[HELP] Hidden")
                print("ℹ️ Help menu toggled OFF")
        elif key == ord("s"):
            # Save drawing
            filename = f"drawing_{int(time.time())}.png"
            saved_path = canvas.save(filename)
            import os
            save_dir = os.path.dirname(saved_path)
            save_filename = os.path.basename(saved_path)
            hud.show(f"[SAVE] {save_filename}")
            print(f"✅ Drawing saved to: {saved_path}")
            print(f"📁 Save directory: {save_dir}")
            print(f"📄 File: {save_filename}")
        elif key >= ord("1") and key <= ord("5"):
            # Quick color selection with number keys (1-5 for 5 colors)
            color_idx = key - ord("1")
            if color_idx < len(menu.colors):
                menu.selected_color_idx = color_idx
                canvas.set_color(menu.get_selected_color())
                hud.show(f"[COLOR] {menu.colors[color_idx][0]}")
                print(f"✅ Color: {menu.colors[color_idx][0]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
