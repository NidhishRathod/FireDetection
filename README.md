# Fire Detection System

This Fire Detection System uses a combination of fire detection, optical flow analysis, edge detection, and texture/contour analysis to detect fire in videos. It can process video files or stream from a webcam.

## Features

- **Fire Detection**: Uses YOLOv5 to detect fire in video streams.
- **Optical Flow Analysis**: Analyzes the motion of pixels in the video to detect any unusual movement, which could indicate fire.
- **Edge Detection**: Utilizes Canny edge detection to enhance fire contours.
- **Texture & Contour Analysis**: Reduces false positives by analyzing the texture and contours in the video frames.
- **User Interface**: A simple GUI that allows users to select a video file or start fire detection using a webcam.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv5
- cvzone
- tkinter (for the GUI)
