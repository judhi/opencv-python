import cv2
import numpy as np
import time
from sklearn.cluster import KMeans  # pip install opencv-python scikit-learn
from scipy.spatial import distance

def get_dominant_color(image, k=1):
    # Resize the image to reduce the number of pixels, speeding up processing
    resized_image = cv2.resize(image, (100, 100))
    # Reshape the image to be a list of pixels
    pixels = resized_image.reshape(-1, 3)
    
    # Use KMeans to find the dominant color
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # The dominant color is the centroid of the largest cluster
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return dominant_color

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        # Capture a single frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        # Get the dimensions of the frame
        height, width, _ = frame.shape
        
        # Calculate the coordinates for the 100x100 center crop
        start_x = width // 2 - 50
        start_y = height // 2 - 50
        cropped_frame = frame[start_y:start_y + 100, start_x:start_x + 100]
        
        # Get the dominant color
        dominant_color = get_dominant_color(cropped_frame, k=1)
        print(dominant_color)

        # Display the captured frame
        cv2.imshow('Captured Image', cropped_frame)
        time.sleep(0.5)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
