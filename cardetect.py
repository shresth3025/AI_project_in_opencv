import numpy as notnp
import cv2 as notcv2
import simpleaudio as notsa
import threading as notthreading

# Specify the full path to the sound file in the E drive
sound_path = r'E:\beep-066.wav'  # Ensure the file extension is .wav

# Load the sound
sound_object = notsa.WaveObject.from_wave_file(sound_path)

# Function to play sound asynchronously
def play_sound():
    sound_object.play()

# Specify the full path to the cascade classifier XML file in the E drive
classifier_path = r'E:\shresth_car.xml'

# Specify the full path to the video file in the E drive
video_path = r'E:\carzzzzz.mp4'

# Create the CascadeClassifier object using the full path
classifier_object = notcv2.CascadeClassifier(classifier_path)

# Open the video file
video_capture = notcv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not video_capture.isOpened():
    print("Error: Unable to open video file")
    exit()
# Loop through the frames of the video
#looping:
"""Looping through frames in video processing involves iterating over each frame of a video file or video stream, performing operations or analyses on each frame as necessary. This process is fundamental in various computer vision and video processing applications, including object detection, tracking, recognition, and video enhancement.

When we say "looping through frames," we are referring to the process of sequentially accessing each frame of a video, one after the other, until the end of the video is reached. Each frame represents a single image captured at a specific moment in time, typically at a fixed frame rate (e.g., 24 frames per second for movies). The frame rate determines how many frames are displayed per second, influencing the smoothness of motion in the video.

The loop starts by opening the video file or initializing the video stream. Once the video is opened, the loop continues until the end of the video or until it is manually interrupted. During each iteration of the loop, a single frame is read from the video source. This frame is then processed or analyzed according to the requirements of the application.

Processing each frame typically involves converting it to a suitable format for analysis, such as converting color images to grayscale or applying various filters or transformations. For example, in object detection applications, the frame may be converted to grayscale to simplify processing, or it may undergo preprocessing steps such as noise reduction or contrast enhancement.

After processing, the frame is often analyzed to detect or track objects of interest. This can be done using techniques such as feature extraction, template matching, or machine learning-based algorithms. For instance, in object detection, the frame may be scanned for specific patterns or shapes that resemble objects in a predefined database, and bounding boxes may be drawn around detected objects for visualization.

Once processing and analysis are complete for a particular frame, the resulting output, such as detected objects or modified frames, can be displayed, saved to disk, or used as input for further processing. In many cases, real-time feedback is provided to the user, allowing them to interact with the video processing application or make decisions based on the analyzed frames.

Throughout the looping process, it's essential to handle any errors or exceptions that may occur, such as corrupted frames or unexpected file formats. Additionally, efficient memory management and resource usage are crucial, especially when processing large video files or streaming high-definition video.

In summary, looping through frames in video processing involves iteratively accessing, processing, analyzing, and visualizing individual frames of a video to extract meaningful information or perform specific tasks. It forms the backbone of many computer vision and video processing applications, enabling tasks ranging from basic frame manipulation to complex object detection and tracking algorithms."""
while video_capture.isOpened():
    try:
        # Read a frame from the video
        ret, frame = video_capture.read()
        
        # Check if the frame is read successfully
        if not ret:
            print("End of video reached")
            break
        
        # Convert the frame to grayscale
        gray_frame = notcv2.cvtColor(frame, notcv2.COLOR_BGR2GRAY)
        
        # Detect objects in the frame
        objects = classifier_object.detectMultiScale(gray_frame, 1.3, 5)
        
        # Draw rectangles around the detected objects with cyan color
        for (x, y, w, h) in objects:
            notcv2.rectangle(frame, (x, y), (x+w+5, y+h+5), (255, 255, 0), 3)
            # Play sound asynchronously
            notthreading.Thread(target=play_sound).start()
        
        # Display the frame with object detections
        notcv2.imshow('Object Detector', frame)
        
        # Check for key press
        key = notcv2.waitKey(30) & 0xff
        if key == 27:  # Press 'Esc' to exit
            break
    
    except Exception as e:
        print("Error:", e)
        continue

# Release the video capture object and close all windows
video_capture.release()
notcv2.destroyAllWindows()
