# AirSketch (Gesture-Controlled Drawing)

This repository contains **AirSketch**, a webcam-based drawing application controlled entirely with hand gestures (MediaPipe Hands + OpenCV).

## 1) Quick Start (Windows)

### Prerequisites

- Windows 10/11
- Python 3.10+ (recommended: 3.11)
- A working webcam

### Setup (from the project root folder)

1. Create & activate a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python airskt\app.py
```

If your camera is in use by another app, close it first.

## 2) Controls / Gestures (In-App)

### Keyboard

- **H**: Toggle help / gesture instructions
- **S**: Save the current drawing to a PNG
- **Q**: Quit

### Gestures

- ☝️ **Index Up**: Freehand draw
- ✌️ **Peace Sign**: Eraser (erase where the cursor is)
- ✊ **Fist**: Pause / Resume
- 👋 **Swipe Left**: Clear canvas
- 👌 **Pinch** (thumb+index close):
  - On top menu bar: select Colors / Brush size / Tools / Shapes
  - On canvas (when a Shape is selected): pinch and RELEASE to draw that shape
- 👍 **Thumbs Up**: Run shape recognition on the current canvas drawing

Note: shape recognition only runs on thumbs up (and is rate-limited by a cooldown).

## 3) Code Structure

### Project root

- `airskt/app.py`  
  Main application loop:
  - Reads webcam frames
  - Runs MediaPipe hand tracking
  - Interprets gestures
  - Updates canvas + renders UI/menu + overlays

### `airskt/core/`

Core modules used by the app:

- `tracker.py`  
  `HandTracker`: wraps MediaPipe Hands initialization and landmark retrieval.

- `gestures.py`  
  `GestureRecognizer`: converts hand landmarks into high-level gestures (thumbs up, pinch, fist, swipe, peace sign, etc).

- `canvas.py`  
  `AirCanvas`: stores the drawing surface and implements:
  - freehand drawing
  - erasing
  - shape drawing

- `menu.py`  
  `PaintMenu`: draws the top menu bar and handles hit-testing for selections.

- `utils.py`  
  `OneEuroFilter` (smoothing) and helpers used for stable drawing motion.

- `hud.py`  
  `HUDMessage`: small on-screen notifications (e.g., selections, shape detected).

### Other

- `COMPLETED_FEATURES.md`  
  Notes about completed features and improvements.

## 4) Running in a Fresh Machine

- Do **NOT** commit/upload the `venv/` folder to GitHub (it is large and machine-specific). This repo includes a `.gitignore` to exclude it.
- Install via `requirements.txt` and run using:

```bash
python airskt\app.py
```

## 5) Common Troubleshooting

- **"Module not found" errors**: Ensure you are running from the project root and you installed requirements:

```bash
python airskt\app.py
```

- **Low FPS / lag**: Use good lighting, reduce background clutter, and ensure no other heavy apps are using the GPU/CPU.
- **Thumbs up not triggering**: Hold a clear thumbs up with other fingers folded; keep your hand visible and steady.
