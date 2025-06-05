from datetime import datetime, timedelta
from controller import Robot, Motion

class NaoMotion:
  
   ms = ["Forwards", "Backwards", "TurnLeft40", "TurnRight40", "SideStepLeft", "SideStepRight"]
   motions = {}
  
   def __init__(self, robot):
       self.robot = robot
       self.timestep = int(robot.getBasicTimeStep())
       self._createMotions() 
  
   def _createMotions(self):
    for n in self.ms: 
        m3 = self._createSpeeds(n)
        self.motions[n] = m3
      
   def _createSpeeds(self, n):
    with open("../../motions/"+n+".motion") as f:
        header = f.readline()
        lines1 = header
        lines2 = header
        lines3 = header
        line = f.readline()
        while line is not None and line != "":
            t = line[:line.index(",")]
            t = datetime.strptime(t, "%M:%S:%f").time()
            t = timedelta(hours=0, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
            ts1 = f"00:{(t*2).seconds:02d}:{int((t*2).microseconds/1000):03d}"
            ts2 = f"00:{t.seconds:02d}:{int(t.microseconds/1000):03d}"
            ts3 = f"00:{int((t/1.8).seconds):02d}:{int((t/1.8).microseconds/1000):03d}"
            lines1 += ts1+line[line.index(","):] 
            lines2 += line
            lines3 += ts3+line[line.index(","):] 
            line = f.readline()
        with open("../../motions/"+n+"_1.motion", "w") as f:
            f.write(lines1)
        with open("../../motions/"+n+"_2.motion", "w") as f:
            f.write(lines2)
        with open("../../motions/"+n+"_3.motion", "w") as f:
            f.write(lines3)
        return (Motion("../../motions/"+n+"_1.motion"), 
                Motion("../../motions/"+n+"_2.motion"), 
                Motion("../../motions/"+n+"_3.motion"))

   def _checkspeed(self, speed):
        if type(speed) != int: raise TypeError("speed should be an integer")
        if speed < 1 or speed > 3: raise ValueError("speed should be 1, 2, 3")
     
   def _checkdir(self, dir):
        if type(dir) != str: raise TypeError("direction should be a string")
        if dir != "left" and dir != "right": raise ValueError("direction should be left or right")
    
   def _applyMotion(self, motion):
        motion.play()
        # print(motion.getDuration())
        while self.robot.step(self.timestep) != -1:
            if motion.isOver() and motion.getTime() >= motion.getDuration() - 5: 
                # print("---", motion.getTime())
                return 
    
   def forward(self, speed):
        self._checkspeed(speed) 
        motion = self.motions["Forwards"][speed-1]
        self._applyMotion(motion)
       
   def backward(self, speed):
        self._checkspeed(speed) 
        motion = self.motions["Backwards"][speed-1]
        self._applyMotion(motion)
    
   def turn(self, dir, speed):
       self._checkdir(dir)
       self._checkspeed(speed)
       if dir=="left": motion = self.motions["TurnLeft40"][speed-1]
       else: motion = self.motions["TurnRight40"][speed-1]
       self._applyMotion(motion)
       
   def sidestep(self, dir, speed):
       self._checkdir(dir)
       self._checkspeed(speed)
       if dir=="left": motion = self.motions["SideStepLeft"][speed-1]
       else: motion = self.motions["SideStepRight"][speed-1]
       self._applyMotion(motion)
