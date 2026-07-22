# 🛣️ Road Image Detection

An AI-powered web app that detects road surface damage — cracks, potholes, and more — directly from photos, using a fine-tuned YOLO object detection model.

🔗 **Live App:** [road-image-detection-kacir8ejzzrkglvvumvvtc.streamlit.app](https://road-image-detection-kacir8ejzzrkglvvumvvtc.streamlit.app/)

## Overview
Road Image Detection helps automate road inspection by identifying common pavement defects in images, making it easier and faster to assess road condition without manual review.

## Features
- 📤 Upload one or multiple images at once
- 🎚️ Adjustable confidence and IoU thresholds from the sidebar
- 🏷️ Filter which damage types to display
- 🎨 Distinct color per damage type with a clear legend
- 🖼️ Side-by-side view of the original and annotated images
- 📋 Detailed results table (damage type, confidence, bounding box coordinates)
- ⬇️ Download button for the annotated image
- 📊 Summary stats (total detections, average time per image) and a chart of damage distribution
- ℹ️ "About" tab explaining each of the seven damage types

## Supported Damage Types
| Type | Description |
|---|---|
| Alligator | Interconnected cracking resembling alligator skin |
| Block | Rectangular cracking patterns |
| Crack | General surface cracking |
| Edge | Cracking near the pavement edge |
| Longitudinal | Cracks running parallel to the road direction |
| Pothole | Localized surface cavities |
| Transverse | Cracks running across the road direction |

## Tech Stack
- **Model:** YOLO (Ultralytics)
- **Frontend:** Streamlit
- **Image Processing:** OpenCV, Pillow, NumPy

## Project Files
```
├── app.py              # Streamlit application
├── best_model.pt        # Trained YOLO weights
├── classes.json          # Damage class names
├── requirements.txt      # Python dependencies
└── README.md
```

