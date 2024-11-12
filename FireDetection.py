# Importing necessary libraries
import cv2
import math
import numpy as np
from ultralytics import YOLO
import cvzone
from tkinter import Tk, Button, filedialog

model = YOLO('fire.pt')  # Ensure 'fire.pt' is in the same directory or provide a full path
classnames = ['fire']


def select_video_file():
    """Opens a file dialog for the user to select a video file and starts fire detection on it."""
    file_path = filedialog.askopenfilename(title="Select a Video File",
                                           filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if file_path:
        detect_fire(file_path)


def open_webcam():
    """Starts fire detection using the webcam."""
    detect_fire(0)  # 0 is the default webcam index


def detect_fire(source):
    """
    Detects fire in the video source (file or webcam).

    Args:
        source: Video file path or webcam index.
    """
    cap = cv2.VideoCapture(source)
    prev_frame = None  # Variable to store the previous frame for optical flow analysis

    while True:
        # Step 2: Read Frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video source has no more frames

        # Step 3: Convert Frame to Grayscale
        frame = cv2.resize(frame, (640, 480))  # Resize frame for consistent processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 4: Apply Optical Flow Analysis
        optical_flow_mask = np.zeros_like(frame)  # Initialize mask for optical flow visualization
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            optical_flow_mask = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            optical_flow_mask = cv2.cvtColor(optical_flow_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        prev_frame = gray  # Update the previous frame for the next iteration

        # Step 5: Apply Edge Detection
        edges = cv2.Canny(gray, 100, 200)  # Canny edge detection

        # Step 6: Perform Texture and Contour Analysis
        texture_contour_mask = analyze_texture_contours(gray)

        # Step 7: Run code for Fire Detection
        result = model(frame, stream=True)

        # Step 8: Process detection results
        fire_detected = False  # Flag to check if fire is detected
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box and label for detected fire
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 30],
                                       scale=1.5, thickness=2)
                    fire_detected = True  # Set the flag to True if fire is detected

        # Step 9: Overlay optical flow mask and texture contour on the frame for visualization
        frame = cv2.addWeighted(frame, 0.5, optical_flow_mask, 0.5, 0)
        frame = cv2.addWeighted(frame, 0.5, cv2.cvtColor(texture_contour_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

        # Display alert if fire is detected
        if fire_detected:
            cvzone.putTextRect(frame, "Alert: Fire Detected!", (50, 50), scale=2, thickness=3, colorR=(0, 0, 255))

        # Display the final frame
        cv2.imshow('Fire Detection', frame)

        # Step 9: Next Frame or End
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def analyze_texture_contours(gray):
    """
    Analyze the texture and contours in the grayscale frame to reduce false positives.

    Args:
        gray: Grayscale image.

    Returns:
        contour_mask: Mask with contours drawn.
    """
    # Perform Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to detect contours more effectively
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray)

    # Draw contours on the mask
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)

    return contour_mask


# Set up the graphical user interface using Tkinter
def setup_gui():
    """Sets up the graphical user interface for selecting video files or opening the webcam."""
    root = Tk()
    root.title("Fire Detection System")
    root.geometry("300x150")

    # Button to select a video file for fire detection
    btn_video = Button(root, text="Select Video File", command=select_video_file, width=20, height=2)
    btn_video.pack(pady=10)

    # Button to start fire detection using the webcam
    btn_webcam = Button(root, text="Open Webcam", command=open_webcam, width=20, height=2)
    btn_webcam.pack(pady=10)

    # Start the Tkinter main loop to keep the GUI running
    root.mainloop()


if __name__ == "__main__":
    setup_gui()  # Run the GUI setup
