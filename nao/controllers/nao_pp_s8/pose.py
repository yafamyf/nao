import mediapipe as mp
import cv2

class Pose:

  def __init__(self, min_detection_confidence, min_tracking_confidence, 
               model_complexity, video_source):
      self.mp_pose = mp.solutions.pose
      self.cap = cv2.VideoCapture(video_source)
      self.min_detection_confidence = min_detection_confidence
      self.min_tracking_confidence = min_tracking_confidence
      self.model_complexity = model_complexity
      self.video_source = video_source
 
  def _toVector(self, landmarks):
    if landmarks is None: return []
    line = []
    for i,l in enumerate(landmarks.landmark):
       line.append(str(l.x))
       line.append(str(l.y))
    return line
    
  def getPose(self):
     with self.mp_pose.Pose(min_detection_confidence=self.min_detection_confidence, 
                            min_tracking_confidence=self.min_tracking_confidence, 
                            model_complexity=self.model_complexity) as pose:
         opened, image = self.cap.read()
         results = pose.process(image)
         vec = self._toVector(results.pose_landmarks)
         return vec
