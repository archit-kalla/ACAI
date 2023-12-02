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
        self.steerAngle = 0

    def from_json(self, json_str):
        json_str = json.loads(json_str)
        self.numberOftyresOut = json_str['numberOftyresOut']
        self.rpms = json_str['rpms']
        self.isInvalidLap = json_str['isInvalidLap']
        self.lapcount = json_str['lapcount']
        self.speedKMH = json_str['speedKMH']
        self.normalizedSplinePosition = json_str['normalizedSplinePosition']
        self.gap = json_str['gap']
        self.laptime = json_str['laptime']
        self.slipAngle = json_str['slipAngle']
        self.worldPosition = json_str['worldPosition']
        self.velvector = json_str['velvector']
        self.carDamaged= json_str['carDamaged']
        self.steerAngle = json_str['steerAngle']
        
    