from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Predicting emotion from an Image
img = cv2.imread('Woman1.jpg')
plt.imshow(img)
plt.show()

## BGR to RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Get the most dominant emotion and its probability
predictions = DeepFace.analyze(img)
dominant_emotion = predictions[0]['dominant_emotion']
dominant_emotion_probability = predictions[0]['emotion'][dominant_emotion]

# Print the results
print(f"Most dominant emotion: {dominant_emotion} (Probability: {dominant_emotion_probability:.2f}%)")

# Drawin Rectangle on the face to focus on specific area in the analysis

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.1,4)
for(x, y, w, h) in faces :
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,0), 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#Showing the dominant emotion in the image

# Define the text
text = f"{dominant_emotion} ({dominant_emotion_probability:.2f}%)"

# Define the font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 0, 255)  # BGR color format (red)

# Get the size of the text to determine the position
text_size = cv2.getTextSize(text, font, 1, 2)[0]
# Center the text horizontally
text_x = (img.shape[1] - text_size[0]) // 2
# Adjust the vertical position as needed
text_y = 50

# Put the text on the image
cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2, cv2.LINE_AA)

# Display the image with text
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Define the video output filename and codec
output_filename = 'emotion_detection_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the video writer
out = None

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Analyze emotions in the frame
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the dominant emotion and its probability
    font = cv2.FONT_HERSHEY_SIMPLEX
    dominant_emotion = results[0]['dominant_emotion']
    dominant_emotion_probability = results[0]['emotion'][dominant_emotion]
    cv2.putText(frame,
                f"Emotion: {dominant_emotion} ({dominant_emotion_probability:.2f}%)",
                (50, 50),
                font, 1,
                (0, 0, 255),
                2,
                cv2.LINE_4)

    # Show the video feed
    cv2.imshow('Emotion Detection', frame)

    # Initialize the video writer if it's not already initialized
    if out is None:
        width = int(cap.get(3))  # Width of the frames in the video
        height = int(cap.get(4))  # Height of the frames in the video
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))

    # Write the current frame to the video file
    out.write(frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the camera and video writer
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
