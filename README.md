Fire Detection System
This Fire Detection System uses a combination of fire detection, optical flow analysis, edge detection, and texture/contour analysis to detect fire in videos. It can process video files or stream from a webcam.

Features
Fire Detection: Uses YOLOv5 to detect fire in video streams.
Optical Flow Analysis: Analyzes the motion of pixels in the video to detect any unusual movement, which could indicate fire.
Edge Detection: Utilizes Canny edge detection to enhance fire contours.
Texture & Contour Analysis: Reduces false positives by analyzing the texture and contours in the video frames.
User Interface: A simple GUI that allows users to select a video file or start fire detection using a webcam.
Requirements
Python 3.x
OpenCV
Ultralytics YOLOv5
cvzone
tkinter (for the GUI)
To install the necessary dependencies, run:

bash
Copy
Edit
pip install -r requirements.txt
Or manually install the required libraries:

bash
Copy
Edit
pip install opencv-python opencv-python-headless numpy ultralytics cvzone
Usage
Run the GUI:
To start the system, simply run the following Python script:

bash
Copy
Edit
python fire_detection.py
Once the GUI opens, you will have two options:

Select Video File: Choose a video file to process.
Open Webcam: Start fire detection using the webcam.
The program will display the video feed and overlay the results:

Bounding boxes around detected fire
Alert message when fire is detected
Optical flow and contour visualizations
Press q to exit the video feed.

Model
The system uses a pre-trained YOLOv5 model (fire.pt) for fire detection. Make sure the model file is in the same directory as the script, or provide the full path to the model.

The model is loaded as follows:

python
Copy
Edit
model = YOLO('fire.pt')
Datasets
The model is trained on a custom dataset, specifically designed for fire detection. You may replace the model with any other trained YOLOv5 model for different detection tasks.

Functions
select_video_file(): Opens a file dialog for selecting a video file.
open_webcam(): Starts fire detection using the default webcam.
detect_fire(source): Processes the video feed, detects fire, and overlays results.
analyze_texture_contours(gray): Applies texture and contour analysis to reduce false positives.
