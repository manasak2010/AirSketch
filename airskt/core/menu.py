import cv2
import numpy as np

class PaintMenu:
    """Paint-style menu with color palette, brush sizes, and tools - HORIZONTAL TOP BAR"""
    
    def __init__(self, menu_height=120, canvas_width=960):
        self.menu_height = menu_height
        self.canvas_width = canvas_width
        
        # Colors: BGR format for OpenCV
        self.colors = [
            ("Red", (0, 0, 255)),
            ("Blue", (255, 0, 0)),
            ("Green", (0, 255, 0)),
            ("Yellow", (0, 255, 255)),
            ("White", (255, 255, 255)),  # Can be used for drawing
        ]
        self.selected_color_idx = 0  # Default: Red
        
        # Brush sizes
        self.brush_sizes = [3, 6, 12, 18]
        self.selected_brush_idx = 1  # Default: Medium (6)
        
        # Tools
        self.tools = ["Draw", "Eraser"]
        self.selected_tool_idx = 0  # Default: Draw
        
        # Shapes
        self.shapes = ["Circle", "Square", "Triangle", "Rectangle"]
        self.selected_shape_idx = None  # None = freehand drawing
        
        # Menu layout - horizontal
        self.color_box_size = 60
        self.color_spacing = 15
        self.brush_circle_radius = 25
        self.brush_spacing = 20
        self.tool_box_width = 100
        self.tool_box_height = 50
        
    def get_selected_color(self):
        """Get currently selected color"""
        return self.colors[self.selected_color_idx][1]
    
    def get_selected_brush_size(self):
        """Get currently selected brush size"""
        return self.brush_sizes[self.selected_brush_idx]
    
    def is_eraser_mode(self):
        """Check if eraser is selected"""
        return self.selected_tool_idx == 1 or self.selected_color_idx == 5
    
    def check_color_selection(self, x, y):
        """Check if a color was selected (returns color index or None)"""
        start_x = 20
        box_size = 45  # Updated to match draw function
        spacing = 10
        start_y = 35
        
        for i in range(len(self.colors)):
            box_x = start_x + i * (box_size + spacing)
            if (box_x <= x <= box_x + box_size and 
                start_y <= y <= start_y + box_size):
                return i
        return None
    
    def check_brush_selection(self, x, y):
        """Check if a brush size was selected (returns brush index or None)"""
        start_x = 20 + len(self.colors) * (45 + 10) + 25
        circle_radius = 12
        start_y = 35
        box_size = 45
        
        for i, size in enumerate(self.brush_sizes):
            center_x = start_x + i * (18 + self.brush_spacing) + 12
            center_y = start_y + box_size // 2
            
            # Check if click is within circle
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= circle_radius + 5:  # Add padding for easier selection
                return i
        return None
    
    def check_tool_selection(self, x, y):
        """Check if a tool was selected (returns tool index or None)"""
        # Brushes end, then tools start
        colors_end = 20 + len(self.colors) * (45 + 10)
        brushes_end = colors_end + 25 + len(self.brush_sizes) * (18 + self.brush_spacing) + 20
        start_x = brushes_end + 10
        start_y = 35
        tool_box_h = 38
        
        for i in range(len(self.tools)):
            box_x = start_x + i * (70 + 8)
            if (box_x <= x <= box_x + 70 and 
                start_y <= y <= start_y + tool_box_h):
                return i
        return None
    
    def check_shape_selection(self, x, y):
        """Check if a shape was selected (returns shape index or None)"""
        colors_end = 20 + len(self.colors) * (45 + 10)
        brushes_end = colors_end + 25 + len(self.brush_sizes) * (18 + self.brush_spacing) + 20
        tools_end = brushes_end + 10 + len(self.tools) * (70 + 8)
        start_x = tools_end + 10
        start_y = 35
        tool_box_h = 38
        
        for i in range(len(self.shapes)):
            box_x = start_x + i * (65 + 6)
            if (box_x <= x <= box_x + 65 and 
                start_y <= y <= start_y + tool_box_h):
                return i
        return None
    
    def get_selected_shape(self):
        """Get currently selected shape name or None"""
        if self.selected_shape_idx is not None:
            return self.shapes[self.selected_shape_idx]
        return None
    
    def draw(self, frame):
        """Draw the menu as a horizontal bar at the top"""
        H, W = frame.shape[:2]
        
        # Draw menu background
        menu_bg = np.zeros((self.menu_height, W, 3), dtype=np.uint8)
        menu_bg[:] = (30, 30, 40)  # Dark gray-blue
        
        # Draw border at bottom
        cv2.line(menu_bg, (0, self.menu_height - 1), (W, self.menu_height - 1), (100, 150, 200), 3)
        
        # Draw title
        cv2.putText(menu_bg, "AIRSKETCH", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw color palette - HORIZONTAL (adjusted for 120px height)
        start_x = 20
        start_y = 35  # Moved up slightly
        box_size = 45  # Slightly smaller to fit labels
        spacing = 10  # Reduced spacing
        
        cv2.putText(menu_bg, "Colors:", (start_x, start_y - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        for i, (name, color) in enumerate(self.colors):
            box_x = start_x + i * (box_size + spacing)
            
            # Draw color box
            cv2.rectangle(menu_bg, (box_x, start_y), 
                         (box_x + box_size, start_y + box_size), 
                         color, -1)
            
            # Draw border (highlighted if selected)
            border_color = (0, 255, 255) if i == self.selected_color_idx else (150, 150, 150)
            border_thickness = 4 if i == self.selected_color_idx else 2
            cv2.rectangle(menu_bg, (box_x, start_y), 
                         (box_x + box_size, start_y + box_size), 
                         border_color, border_thickness)
            
            # Draw label below (within menu bounds)
            label_y = start_y + box_size + 10
            if label_y < self.menu_height - 5:  # Ensure label fits
                cv2.putText(menu_bg, name, (box_x + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw brush sizes - HORIZONTAL with larger, more visible circles
        brush_start_x = start_x + len(self.colors) * (box_size + spacing) + 25
        cv2.putText(menu_bg, "Brush:", (brush_start_x, start_y - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        for i, size in enumerate(self.brush_sizes):
            center_x = brush_start_x + i * (18 + self.brush_spacing) + 12
            center_y = start_y + box_size // 2
            
            # Draw larger circle background for visibility
            bg_radius = 12
            border_color = (0, 255, 255) if i == self.selected_brush_idx else (100, 100, 100)
            border_thickness = 3 if i == self.selected_brush_idx else 2
            
            # Background circle
            cv2.circle(menu_bg, (center_x, center_y), bg_radius, 
                      (50, 50, 60), -1)
            # Border circle
            cv2.circle(menu_bg, (center_x, center_y), bg_radius, 
                      border_color, border_thickness)
            # Actual brush size circle (white) - make it more visible
            brush_radius = max(size // 2, 2)
            cv2.circle(menu_bg, (center_x, center_y), brush_radius, 
                      (255, 255, 255), -1)
            
            # Draw label below (within bounds)
            label_y = start_y + box_size + 10
            if label_y < self.menu_height - 5:
                cv2.putText(menu_bg, f"{size}", (center_x - 8, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw tools - HORIZONTAL
        colors_end = start_x + len(self.colors) * (box_size + spacing)
        brushes_end = brush_start_x + len(self.brush_sizes) * (18 + self.brush_spacing) + 20
        tool_start_x = brushes_end + 10
        
        cv2.putText(menu_bg, "Tools:", (tool_start_x, start_y - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        tool_box_h = 38  # Adjusted for 120px menu
        for i, tool in enumerate(self.tools):
            box_x = tool_start_x + i * (70 + 8)
            
            # Draw tool button
            bg_color = (80, 120, 160) if i == self.selected_tool_idx else (60, 60, 70)
            cv2.rectangle(menu_bg, (box_x, start_y), 
                         (box_x + 70, start_y + tool_box_h), 
                         bg_color, -1)
            
            # Draw border
            border_color = (0, 255, 255) if i == self.selected_tool_idx else (100, 100, 100)
            cv2.rectangle(menu_bg, (box_x, start_y), 
                         (box_x + 70, start_y + tool_box_h), 
                         border_color, 3)
            
            # Draw label - Reduced thickness for better clarity
            text_y = start_y + tool_box_h // 2 + 5
            cv2.putText(menu_bg, tool, (box_x + 8, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw shapes - HORIZONTAL
        tools_end = tool_start_x + len(self.tools) * (70 + 8)
        shape_start_x = tools_end + 10
        
        cv2.putText(menu_bg, "Shapes:", (shape_start_x, start_y - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        for i, shape_name in enumerate(self.shapes):
            box_x = shape_start_x + i * (65 + 6)
            
            # Draw shape button
            bg_color = (120, 80, 160) if i == self.selected_shape_idx else (60, 60, 70)
            cv2.rectangle(menu_bg, (box_x, start_y), 
                         (box_x + 65, start_y + tool_box_h), 
                         bg_color, -1)
            
            # Draw border
            border_color = (0, 255, 255) if i == self.selected_shape_idx else (100, 100, 100)
            cv2.rectangle(menu_bg, (box_x, start_y), 
                         (box_x + 65, start_y + tool_box_h), 
                         border_color, 3)
            
            # Draw label (abbreviated to fit) - Reduced thickness for clarity
            shape_label = shape_name[:4]  # First 4 chars: "Circ", "Squa", "Tria", "Rect"
            text_y = start_y + tool_box_h // 2 + 5
            cv2.putText(menu_bg, shape_label, (box_x + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw instructions on right side
        inst_x = W - 180
        cv2.putText(menu_bg, "S=Save | Q=Quit", (inst_x, start_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Overlay menu on frame at top
        frame[:self.menu_height, :] = menu_bg
        return frame
