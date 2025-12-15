import time
import math
import numpy as np
import mediapipe as mp

class GestureRecognizer:
    """
    Minimal + stable:
      - 'fist'       : pause/resume when all fingers folded (hold >= 0.35s, cooldown 0.9s)
      - 'sweep_left' : clear canvas on fast leftward motion (short motion history)
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands

        # motion
        self.prev_centroid = None
        self.prev_time = time.time()
        self.motion_hist = []              # (t, x) recent 300ms

        # swipe params - Made more lenient for better responsiveness
        self.sweep_cooldown = 1.0          # Reduced from 1.5s for faster response
        self.last_sweep_time = 0.0
        self.min_disp_frac = 0.20          # Reduced from 0.25 - must move ~20% of width
        self.min_vx_frac = -0.8             # Reduced from -1.2 - must move faster (vx < -0.8*W per second)
        self.min_sweep_speed_frac = 0.20   # Reduced from 0.3 - minimum speed (20% of width per second)

        # fist (pause) state - IMPROVED
        self.fist_hold_start = None
        self.fist_active = False
        self.last_fist_time = 0.0
        self.fist_hold_time = 0.20         # Reduced from 0.35s for faster response
        self.fist_cooldown = 0.5           # Reduced from 0.9s for faster response
        
        # Rolling average for finger count (3-frame smoothing)
        self.finger_count_history = []     # Last 3 finger counts
        self.finger_history_size = 3
        self.max_fist_fingers = 1.2        # Smoothed finger threshold for fist
        self.fist_confirm_frames = 0       # Require consecutive frames before hold
        self.required_fist_frames = 3
        
        # Thumbs Up (Shape Recognition)
        self.thumbs_up_hold_start = None
        self.last_thumbs_up_time = 0.0
        self.thumbs_up_hold_time = 0.1     # Hold for 0.1s (faster)
        self.thumbs_up_cooldown = 1.5      # Cooldown 1.5s (faster)
        self.thumbs_up_confirm_frames = 0
        
        # Z-depth filtering (hand must be close enough to camera)
        self.max_z_depth = 0.15            # Maximum Z-depth (negative = closer to camera)
        self.min_z_depth = -0.3            # Minimum Z-depth

        # to ignore too-small hands (far from camera)
        self.min_bbox_frac_w = 0.10        # Reduced from 0.12 for better detection

    # ---------- helpers ----------
    def _centroid(self, hand, shape):
        H, W, _ = shape
        cx = sum(lm.x for lm in hand.landmark) / len(hand.landmark)
        cy = sum(lm.y for lm in hand.landmark) / len(hand.landmark)
        return (cx * W, cy * H)

    def _bbox(self, hand, shape):
        H, W, _ = shape
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        x1, x2 = min(xs) * W, max(xs) * W
        y1, y2 = min(ys) * H, max(ys) * H
        return x1, y1, x2, y2

    def _fingers_up(self, hand):
        """
        Count extended fingers using landmark geometry (mirrored webcam).
        Thumb: compare tip(4) vs ip(3) x-positions, with a simple handedness guess.
        Other fingers: tip above pip (y smaller).
        Returns: (total_fingers, individual_finger_states)
        """
        lm = hand.landmark

        # handedness guess: if index MCP (5) is left of wrist (0) -> "right hand" in mirrored view
        right = lm[5].x < lm[0].x

        fingers = 0
        finger_states = {}

        # Thumb (use x because palm faces camera; mirrored image)
        thumb_extended = (lm[4].x < lm[3].x) if right else (lm[4].x > lm[3].x)
        finger_states['thumb'] = thumb_extended
        if thumb_extended:
            fingers += 1

        # Index, Middle, Ring, Pinky: tip above PIP
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        names = ['index', 'middle', 'ring', 'pinky']
        for t, p, name in zip(tips, pips, names):
            extended = lm[t].y < lm[p].y
            finger_states[name] = extended
            if extended:
                fingers += 1

        return fingers, finger_states
    
    def _is_writing_pose(self, hand):
        """
        Check if hand is in writing pose (like React code, but more lenient):
        - Index finger up (tip < pip) - REQUIRED
        - At least 2 of (Middle, Ring, Pinky) down (tip > pip)
        This is the natural drawing position.
        More lenient than strict React version to allow natural hand variations.
        """
        lm = hand.landmark
        # Index up - REQUIRED
        index_up = lm[8].y < lm[6].y
        if not index_up:
            return False
        
        # At least 2 of the other 3 fingers should be down (more lenient)
        fingers_down_count = sum([
            lm[12].y > lm[10].y,  # middle down
            lm[16].y > lm[14].y,  # ring down
            lm[20].y > lm[18].y   # pinky down
        ])
        
        return fingers_down_count >= 2
    
    def _is_thumbs_up(self, hand, finger_states):
        """
        Detect Thumbs Up gesture:
        - Thumb extended
        - All other fingers folded
        """
        if not finger_states['thumb']:
            return False
            
        # Check others are folded
        others_folded = (not finger_states['index'] and 
                         not finger_states['middle'] and 
                         not finger_states['ring'] and 
                         not finger_states['pinky'])
                         
        return others_folded

    def _is_pinch_gesture(self, hand):
        """
        Detect pinch gesture (thumb and index finger close together).
        Used for selecting colors/brushes in menu.
        """
        lm = hand.landmark
        # Get thumb tip (4) and index tip (8)
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        
        # Calculate distance
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        # Pinch if distance is small (threshold ~0.05 normalized)
        return distance < 0.05
    
    def _is_two_finger_gesture(self, hand):
        """
        Detect two-finger gesture (index and middle fingers up, others down).
        Used for eraser tool - much more natural than pinch.
        """
        lm = hand.landmark
        
        # Check if index and middle fingers are up (extended)
        index_up = lm[8].y < lm[6].y  # Index tip above index PIP
        middle_up = lm[12].y < lm[10].y  # Middle tip above middle PIP
        
        # Check if ring and pinky are down (more lenient)
        ring_down = lm[16].y > lm[14].y  # Ring tip below ring PIP
        pinky_down = lm[20].y > lm[18].y  # Pinky tip below pinky PIP
        
        # Thumb should not interfere - check it's not extended towards other fingers
        thumb_not_extended = lm[4].y > lm[2].y  # Thumb tip below thumb IP
        
        # Two fingers up gesture (like peace sign / victory sign)
        # Be lenient - just need 2 fingers up and at least one finger down
        is_peace = index_up and middle_up and (ring_down or pinky_down)
        
        # Debug output
        if is_peace:
            import time
            if not hasattr(self, '_last_peace_debug') or time.time() - self._last_peace_debug > 0.5:
                print(f"✌️ PEACE SIGN DETECTED: index_up={index_up}, middle_up={middle_up}, ring_down={ring_down}, pinky_down={pinky_down}")
                self._last_peace_debug = time.time()
        
        return is_peace

    # ---------- main ----------
    def detect_gesture(self, results, frame_shape):
        H, W, _ = frame_shape
        now = time.time()

        if not results.multi_hand_landmarks:
            self.prev_centroid = None
            self.motion_hist.clear()
            self.fist_hold_start = None
            self.fist_active = False
            return None, {"is_writing": False}

        hand = results.multi_hand_landmarks[0]

        # ignore tiny/unstable hands (too far)
        x1, y1, x2, y2 = self._bbox(hand, frame_shape)
        if (x2 - x1) < self.min_bbox_frac_w * W:
            # too small -> don't emit gestures (prevents random toggles)
            self.prev_centroid = None
            self.motion_hist.clear()
            self.fist_hold_start = None
            self.fist_active = False
            return None, {"is_writing": False}

        # centroid for motion
        cx, cy = self._centroid(hand, frame_shape)

        # ---- Check for writing pose first (needed for both sweep and fist detection) ----
        is_writing = self._is_writing_pose(hand)

        # ---- SWIPE LEFT with short history ----
        self.motion_hist.append((now, cx))
        self.motion_hist = [it for it in self.motion_hist if now - it[0] <= 0.30]  # ~300ms window

        gesture = None

        if len(self.motion_hist) >= 2:
            t0, x0 = self.motion_hist[0][0], self.motion_hist[0][1]
            t1, x1h = self.motion_hist[-1][0], self.motion_hist[-1][1]
            dt = max(t1 - t0, 1e-3)
            disp = x0 - x1h                     # +ve for leftward net motion

            # instantaneous vx using prev centroid (helps kill false triggers)
            if self.prev_centroid is None:
                vx = 0.0
            else:
                vx = (cx - self.prev_centroid[0]) / max(now - self.prev_time, 1e-3)

            min_disp = self.min_disp_frac * W
            min_vx = self.min_vx_frac * W
            speed = abs(disp / dt) if dt > 0 else 0  # Calculate actual speed

            # SWEEP DETECTION: Only trigger if:
            # 1. Large enough displacement (20% of width - more lenient)
            # 2. Fast enough velocity (moving left quickly)
            # 3. Fast enough speed (deliberate movement - 20% of width per second - more lenient)
            # 4. Cooldown passed
            # 5. NOT in writing pose (can't clear while drawing)
            min_speed = self.min_sweep_speed_frac * W * 0.67  # Reduce to 20% of width per second (was 30%)
            min_disp_relaxed = self.min_disp_frac * W * 0.8  # Reduce to 20% of width (was 25%)
            if (disp > min_disp_relaxed and 
                vx < min_vx and 
                speed > min_speed and
                (now - self.last_sweep_time) > self.sweep_cooldown and
                not is_writing):  # Prevent sweep during drawing
                self.last_sweep_time = now
                gesture = "sweep_left"
                print(f"[SWEEP] disp={disp:.0f}, vx={vx:.0f}, speed={speed:.0f}")
        
        # ---- FIST (all fingers folded) with hold + cooldown + Z-depth filtering ----
        fingers, finger_states = self._fingers_up(hand)
        
        # Check Z-depth (using wrist landmark z-coordinate)
        wrist_z = hand.landmark[0].z  # Wrist landmark
        
        # Add finger count to rolling average
        self.finger_count_history.append(fingers)
        if len(self.finger_count_history) > self.finger_history_size:
            self.finger_count_history.pop(0)
        
        # Use smoothed finger count (average of last 3 frames)
        smoothed_fingers = (sum(self.finger_count_history) / len(self.finger_count_history)) if self.finger_count_history else float(fingers)
        
        # Check if hand is at appropriate depth (not too far, not too close)
        z_valid = self.min_z_depth <= wrist_z <= self.max_z_depth
        z_warning = not z_valid  # Track if Z-depth is out of range for warning

        # ---- STRICT FIST DETECTION: All fingers must be down (fingers == 0) ----
        # A true fist = ALL fingers down, NOT in writing pose
        # This is more reliable than lenient detection
        is_fist = (fingers == 0 and smoothed_fingers <= 0.5) and not is_writing
        
        # ---- FIST DETECTION with confirmation frames ----
        if is_fist:
            self.fist_confirm_frames += 1
        else:
            self.fist_confirm_frames = 0
            self.fist_hold_start = None

        # Require 2 consecutive frames for confirmation
        if self.fist_confirm_frames >= 2:
            # if just started holding, mark start
            if not self.fist_hold_start:
                self.fist_hold_start = now
            # if held long enough and cooldown passed, trigger once
            elif (now - self.fist_hold_start) >= self.fist_hold_time and (now - self.last_fist_time) >= self.fist_cooldown:
                gesture = "fist"
                self.last_fist_time = now
                self.fist_hold_start = None
                self.fist_confirm_frames = 0
                if z_warning:
                    print(f"⚠️ Fist detected but Z-depth out of range: {wrist_z:.3f}")

        # ---- THUMBS UP DETECTION (Shape Recognition) ----
        is_thumbs_up = self._is_thumbs_up(hand, finger_states)
        
        if is_thumbs_up:
            self.thumbs_up_confirm_frames += 1
        else:
            self.thumbs_up_confirm_frames = 0
            self.thumbs_up_hold_start = None
            
        if self.thumbs_up_confirm_frames >= 2:
            if not self.thumbs_up_hold_start:
                self.thumbs_up_hold_start = now
            elif (now - self.thumbs_up_hold_start) >= self.thumbs_up_hold_time and (now - self.last_thumbs_up_time) >= self.thumbs_up_cooldown:
                gesture = "thumbs_up"
                self.last_thumbs_up_time = now
                self.thumbs_up_hold_start = None
                self.thumbs_up_confirm_frames = 0
                print("👍 Thumbs Up Detected!")

        self.prev_centroid = (cx, cy)
        self.prev_time = now

        # ---------- DEBUG: print finger states for fist detection ----------
        if fingers is not None:
            # Print when fist is being detected (even if not yet triggered)
            # Reduced debug spam - only print every 10 frames or if changed
            if is_fist and self.fist_confirm_frames > 2 and self.fist_confirm_frames % 5 == 0:
                 pass # Squelch debug logs
            # Print when gesture is detected
            if gesture:
                print(f"[GESTURE] {gesture} | Fingers: {fingers} (smoothed: {smoothed_fingers:.1f})")

        # Check for pinch gestures
        is_pinch = self._is_pinch_gesture(hand)
        is_two_finger = self._is_two_finger_gesture(hand)
        
        # Return gesture and status info
        return gesture, {
            "is_writing": is_writing,
            "is_pinch": is_pinch,
            "is_two_finger": is_two_finger,
            "hand_landmarks": hand.landmark if results.multi_hand_landmarks else None
        }
