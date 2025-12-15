import mediapipe as mp

class HandTracker:
    def __init__(self, detection_conf=0.6, tracking_conf=0.6):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # 0=Faster/Lite, 1=Default. Using 0 for smoother FPS
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def get_index_fingertip(self, frame):
        """Returns (x, y) of index fingertip or None."""
        rgb = frame[:, :, ::-1]
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                lm = hand.landmark[8]  # index fingertip
                h, w, _ = frame.shape
                return int(lm.x * w), int(lm.y * h)
        return None
