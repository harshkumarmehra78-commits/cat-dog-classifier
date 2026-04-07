import cv2
import numpy as np

def extract_frames(video_path, max_frames=10, size=(32, 32)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (32,32))   # MUST be 32x32
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Padding if frames are less
    while len(frames) < max_frames:
        frames.append(np.zeros((32,32,3)))

    return np.array(frames)