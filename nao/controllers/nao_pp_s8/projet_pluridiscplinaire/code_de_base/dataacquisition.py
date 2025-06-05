import cv2
import mediapipe as mp

"""
This script captures video from the webcam and uses MediaPipe to detect human pose landmarks.
It draws the detected landmarks and connections on the video feed.
Press 'q' or 'esc' to exit the video feed.
Press 'a' to save the landmarks to a file.
Parameters are given below, including the filename where to save the results (csv, with X,Y coordinates of landmarks).
"""

# parameters
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1
video_source = 0  # Default webcam
output_file = "landmarks.csv" 
# end parameters


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_source)

landmarks_c= (234,63,247)
connection_c= 240 #(117,249,77), 
thickness=3
circle_r=2
opened=True
tosave = []

def keepLandmarks(landmarks, tosave):
    if len(tosave) == 0:
      header = []
      for i,l in enumerate(landmarks.landmark):
         header.append(f"{mp_pose.PoseLandmark(i).name}_X")
         header.append(f"{mp_pose.PoseLandmark(i).name}_Y")
      tosave.append(header)
    line = []
    for i,l in enumerate(landmarks.landmark):
       line.append(str(l.x))
       line.append(str(l.y))
    tosave.append(line)
    print("added to list of landmarks to save")

def saveLandmarks(tosave, fn):
   with open(fn, "w") as f:
      for line in tosave:
         f.write(",".join(line) + "\n")
   print("save landmarks to", fn)
  
with mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                  min_tracking_confidence=min_tracking_confidence, 
                  model_complexity=model_complexity) as pose:
  while opened:
     opened, image = cap.read()
     results = pose.process(image)
     output_img = image.copy()
     if results.pose_landmarks:
        mp_drawing.draw_landmarks(output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                    mp_drawing.DrawingSpec(connection_c, thickness, circle_r))
     cv2.imshow("Media pipe pose detection", output_img)
     key = cv2.waitKey(10)
     if key & 0xFF == ord("a"): 
        keepLandmarks(results.pose_landmarks, tosave)
     if key & 0xFF == ord("q"): break
     if key == 27: break

cap.release()
cv2.destroyAllWindows()
saveLandmarks(tosave, output_file)