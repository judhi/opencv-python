# Detecting face on webcam
# and send the center coordinate
# of the face to Serial Port
# by: Judhi P. Nov 2024

import cv2
import serial
import time

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define serial port for communication
ser = serial.Serial('COM5', 9600)  # change with your own serial port

# Initialize video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get the dimensions of the current frame
        frame_height, frame_width = frame.shape[:2]
        
        # Convert frame to grayscale (Haar cascades work on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Loop over the detected faces
        for (x, y, w, h) in faces:
            # Calculate the center of the face
            centerX = x + w // 2
            centerY = y + h // 2

            # Convert center coordinates to percentage
            centerX_percent = int((centerX / frame_width) * 100)
            centerY_percent = int((centerY / frame_height) * 100)
                
            # Draw bounding box and center point
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)
          
            cv2.putText(frame, f"Face ({centerX_percent}, {centerY_percent})", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Send center coordinates over serial
            print(f"{centerX_percent},{centerY_percent}\n")
            ser.write(f"{centerX_percent},{centerY_percent}\n".encode())
            time.sleep(0.1)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted")

finally:
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
