import time
import cv2
import numpy as np


class HUDMessage:
    def __init__(self):
        self.active_msg = None
        self.start_time = 0
        self.duration = 2.0  # Increased duration for better visibility

    def show(self, msg):
        self.active_msg = msg
        self.start_time = time.time()

    def _draw_rounded_rect(self, img, pt1, pt2, color, thickness=-1, radius=10):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw main rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw rounded corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

    def draw(self, frame, position="bottom_right"):
        """Draw HUD message at specified position (default: bottom_right)"""
        if self.active_msg is None:
            return frame

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active_msg = None
            return frame

        # Fade out effect
        alpha = 1 - (elapsed / self.duration)
        alpha = max(0, min(1, alpha))
        
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = self.active_msg
        
        # Calculate text size
        scale, thickness = 1.0, 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Padding for rounded box
        padding = 15
        H, W = frame.shape[:2]
        
        # Position in bottom right corner
        box_x2 = W - 20
        box_y2 = H - 20
        box_x1 = box_x2 - text_width - padding * 2
        box_y1 = box_y2 - text_height - padding * 2
        
        # Draw semi-transparent rounded box background
        box_overlay = overlay.copy()
        bg_color = (20, 20, 40)  # Dark blue-gray
        self._draw_rounded_rect(box_overlay, (box_x1, box_y1), (box_x2, box_y2), 
                               bg_color, thickness=-1, radius=12)
        cv2.addWeighted(box_overlay, 0.9 * alpha, overlay, 1 - 0.9 * alpha, 0, overlay)
        
        # Draw border
        border_color = (100, 200, 255)  # Light blue
        self._draw_rounded_rect(overlay, (box_x1, box_y1), (box_x2, box_y2), 
                               border_color, thickness=2, radius=12)
        
        # Draw text with shadow effect
        text_x = box_x1 + padding
        text_y = box_y1 + text_height + padding - 5
        
        # Shadow
        cv2.putText(overlay, text, (text_x + 2, text_y + 2), font, scale, 
                   (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Main text
        text_color = (100, 255, 255)  # Cyan
        cv2.putText(overlay, text, (text_x, text_y), font, scale, 
                   text_color, thickness, cv2.LINE_AA)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
