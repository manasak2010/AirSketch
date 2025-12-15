import numpy as np
import cv2

class AirCanvas:
    def __init__(self, width=1280, height=720, color=(0, 0, 255), thickness=6):
        self.width = width
        self.height = height
        self.color = color  # Default: Red
        self.thickness = thickness
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None
        self.eraser_mode = False
        # Store drawn shapes for resizing
        self.shapes = []  # List of dicts: {type, center, size, color, thickness}

    def set_color(self, color):
        """Set drawing color"""
        self.color = color
        # Only set eraser mode if explicitly in eraser tool, not just white color
        # White can be a valid drawing color
        self.eraser_mode = False

    def set_thickness(self, thickness):
        """Set brush thickness"""
        self.thickness = thickness

    def draw_point(self, point):
        if self.prev_point is None:
            self.prev_point = point
            return
        
        if self.eraser_mode:
            # Eraser: draw with black (background color) using brush thickness
            cv2.line(self.image, self.prev_point, point, (0, 0, 0), self.thickness, cv2.LINE_AA)
        else:
            # Normal drawing - ensure color is visible
            # For black on black background, use a lighter gray so it's visible
            if self.color == (0, 0, 0):
                # Use medium gray (40,40,40) so it's clearly visible on black background
                draw_color = (40, 40, 40)
            else:
                draw_color = self.color
            cv2.line(self.image, self.prev_point, point, draw_color, self.thickness, cv2.LINE_AA)
        self.prev_point = point

    def erase_point(self, point, radius=25):
        """Erase at point (for two-finger eraser) - increased default size"""
        # Draw black circle to erase (always erase, regardless of prev_point)
        cv2.circle(self.image, point, radius, (0, 0, 0), -1)
        # If prev_point exists, also erase along the line to prevent gaps
        if self.prev_point is not None:
            # Draw a thick black line between prev and current point for smooth erasing
            cv2.line(self.image, self.prev_point, point, (0, 0, 0), radius * 2, cv2.LINE_AA)
        self.prev_point = point

    def clear(self):
        self.image = np.zeros_like(self.image)
        self.prev_point = None
        self.shapes = []  # Clear all stored shapes

    def draw_shape(self, shape_type, center, size=50, store=True):
        """Draw a predefined shape at center point. If store=True, adds to shapes list for resizing."""
        if self.eraser_mode:
            draw_color = (0, 0, 0)  # Black for eraser
        else:
            if self.color == (0, 0, 0):
                draw_color = (40, 40, 40)  # Medium gray for visibility
            else:
                draw_color = self.color
        
        # Store shape if requested (for resizing)
        if store:
            shape_info = {
                'type': shape_type,
                'center': center,
                'size': size,
                'color': draw_color,
                'thickness': self.thickness
            }
            self.shapes.append(shape_info)
        
        # Draw the shape
        self._draw_shape_impl(shape_type, center, size, draw_color, self.thickness)
    
    def _draw_shape_impl(self, shape_type, center, size, draw_color, thickness):
        """Internal method to draw a shape without storing it"""
        if shape_type == "Circle":
            cv2.circle(self.image, center, size, draw_color, thickness, cv2.LINE_AA)
        elif shape_type == "Square":
            half_size = size // 2
            pt1 = (center[0] - half_size, center[1] - half_size)
            pt2 = (center[0] + half_size, center[1] + half_size)
            cv2.rectangle(self.image, pt1, pt2, draw_color, thickness, cv2.LINE_AA)
        elif shape_type == "Triangle":
            # Equilateral triangle
            h = int(size * 0.866)  # Height of equilateral triangle
            pt1 = (center[0], center[1] - h)
            pt2 = (center[0] - size // 2, center[1] + h // 2)
            pt3 = (center[0] + size // 2, center[1] + h // 2)
            pts = np.array([pt1, pt2, pt3], np.int32)
            cv2.polylines(self.image, [pts], True, draw_color, thickness, cv2.LINE_AA)
        elif shape_type == "Rectangle":
            # Rectangle with 2:1 aspect ratio (wider than tall)
            w, h = int(size * 1.5), size  # Width is 1.5x the size, height is the size
            pt1 = (center[0] - w // 2, center[1] - h // 2)
            pt2 = (center[0] + w // 2, center[1] + h // 2)
            cv2.rectangle(self.image, pt1, pt2, draw_color, thickness, cv2.LINE_AA)
    
    def redraw_all_shapes(self, preserve_image=None):
        """Redraw all stored shapes. If preserve_image is provided, shapes are drawn on top of it."""
        if preserve_image is not None:
            # Draw shapes on top of existing image (preserves freehand drawings)
            temp_image = preserve_image.copy()
            for shape in self.shapes:
                self._draw_shape_on_image(temp_image, shape['type'], shape['center'], shape['size'], 
                                         shape['color'], shape['thickness'])
            self.image = temp_image
        else:
            # Clear and redraw (original behavior)
            self.image = np.zeros_like(self.image)
            for shape in self.shapes:
                self._draw_shape_impl(shape['type'], shape['center'], shape['size'], 
                                    shape['color'], shape['thickness'])
    
    def _draw_shape_on_image(self, image, shape_type, center, size, draw_color, thickness):
        """Draw a shape on a specific image (for redrawing)"""
        if shape_type == "Circle":
            cv2.circle(image, center, size, draw_color, thickness, cv2.LINE_AA)
        elif shape_type == "Square":
            half_size = size // 2
            pt1 = (center[0] - half_size, center[1] - half_size)
            pt2 = (center[0] + half_size, center[1] + half_size)
            cv2.rectangle(image, pt1, pt2, draw_color, thickness, cv2.LINE_AA)
        elif shape_type == "Triangle":
            h = int(size * 0.866)
            pt1 = (center[0], center[1] - h)
            pt2 = (center[0] - size // 2, center[1] + h // 2)
            pt3 = (center[0] + size // 2, center[1] + h // 2)
            pts = np.array([pt1, pt2, pt3], np.int32)
            cv2.polylines(image, [pts], True, draw_color, thickness, cv2.LINE_AA)
        elif shape_type == "Rectangle":
            # Rectangle with 2:1 aspect ratio (wider than tall)
            w, h = int(size * 1.5), size  # Width is 1.5x the size, height is the size
            pt1 = (center[0] - w // 2, center[1] - h // 2)
            pt2 = (center[0] + w // 2, center[1] + h // 2)
            cv2.rectangle(image, pt1, pt2, draw_color, thickness, cv2.LINE_AA)
    
    def resize_last_shape(self, new_size):
        """Resize the most recently drawn shape. Clears canvas and redraws all shapes with new size."""
        if self.shapes:
            # Clamp the new size
            clamped_size = max(10, min(new_size, 500))
            old_size = self.shapes[-1]['size']
            
            # Only update if size actually changed
            if abs(clamped_size - old_size) < 1:
                return False
                
            # Update the size
            self.shapes[-1]['size'] = clamped_size
            
            # Clear canvas and redraw all shapes with updated sizes
            # This will erase freehand drawings, but resize will work correctly
            self.image = np.zeros_like(self.image)
            for shape in self.shapes:
                self._draw_shape_impl(shape['type'], shape['center'], shape['size'], 
                                    shape['color'], shape['thickness'])
            
            return True
        return False
    
    def get_last_shape(self):
        """Get the most recently drawn shape info"""
        return self.shapes[-1] if self.shapes else None
    
    def save(self, filename="drawing.png"):
        """Save canvas to file"""
        import os
        full_path = os.path.abspath(filename)
        cv2.imwrite(full_path, self.image)
        return full_path
