from fer import Video
from fer import FER
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

#Source of the video
Video_source = "emotion_detection_output.mp4"

#Build the Face detection
face_detector = FER(mtcnn=True)

#Video for processing
input_video = Video(Video_source)
processing_video = input_video.analyze(face_detector,display=False)

#Building a Dataframe
vid_df = input_video.to_pandas(processing_video)
vid_df = input_video.get_first_face(vid_df)
vid_df = input_video.get_emotions(vid_df)

#Vizualisation
pltfig = vid_df.plot(figsize=(20,8), fontsize=16).get_figure()
plt.show()

#Sum of each emotion
angry = sum(vid_df.angry)
disgust = sum(vid_df.disgust)
fear = sum(vid_df.fear)
happy = sum(vid_df.happy)
sad = sum(vid_df.sad)
surprise = sum(vid_df.surprise)
neutral = sum(vid_df.neutral)

# Total number of frames
total_frames = len(vid_df)

# Calculate the proportions
angry_percentage = (angry / total_frames) * 100
disgust_percentage = (disgust / total_frames) * 100
fear_percentage = (fear / total_frames) * 100
happy_percentage = (happy / total_frames) * 100
sad_percentage = (sad / total_frames) * 100
surprise_percentage = (surprise / total_frames) * 100
neutral_percentage = (neutral / total_frames) * 100

#Emotions
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
emotions_values = [angry_percentage, disgust_percentage, fear_percentage, happy_percentage, sad_percentage, surprise_percentage, neutral_percentage]


#Score
emotion_proportions = pd.DataFrame(emotions, columns=['Emotion'])
emotion_proportions['Proportion (%)'] = emotions_values
print(emotion_proportions)

# Plot the proportions as a bar graph
plt.figure(figsize=(20, 8))
plt.bar(emotion_proportions['Emotion'], emotion_proportions['Proportion (%)'], color=['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange'])
plt.xlabel('Emotion')
plt.ylabel('Proportion (%)')
plt.title('Emotion Proportions in Video')
plt.xticks(rotation=45)

# Show the bar graph
plt.show()

# Plot the proportions as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(emotion_proportions['Proportion (%)'], labels=emotion_proportions['Emotion'], autopct='%1.1f%%', startangle=140)
plt.title('Emotion Proportions in Video (Pie Chart)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.show()

