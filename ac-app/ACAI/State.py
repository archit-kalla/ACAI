import json
class State:
    
    def __init__(self):
        self.numberOftyresOut = 0
        self.rpms = 0
        self.isInvalidLap = 0
        self.lapcount = 0
        self.speedKMH = 0
        self.normalizedSplinePosition = 0
        self.gap = 0
        self.laptime = 0
        self.slipAngle = 0
        self.worldPosition = [0,0,0]
        self.velvector = [0,0,0]
        self.carDamaged= 0
        self.session_time_left = 0
        self.steerAngle = 0
    
    #json
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    
    