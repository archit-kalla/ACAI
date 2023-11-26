import sys
import ac
import acsys 
import os
import platform
import requests
import json
import threading
import time
import mmap
import math


if platform.architecture()[0] == "64bit":
  sysdir = "stdlib64"
else:
  sysdir = "stdlib"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), sysdir))
os.environ['PATH'] = os.environ['PATH'] + ";."

from sim_info import info
import State
l_lapcount, l_validlap = 0, 0
lapcount, numberOftyresOut = 0, 0
l_rpms = 0
rpms = 0
isInvalidLap = False
state=None
t= None
stopped = False
# lock = threading.Lock()
prelap = False
i=0
prelaptime= True
with open("bestLapMap.json", 'r') as f:
    data = json.load(f)
    bestLapMap = data["arr"]
    storedLapTime = data["time"]
    f.close()
currlapMap =[]
def acMain(ac_version):
    global state, t
    global l_lapcount, l_validlap, l_rpms, l_speedKMH, l_normalizedSplinePosition, l_gap, l_laptime, l_slipAngle, l_session_time_left
    
    appwindow = ac.newApp("ACAI")
    ac.setSize(appwindow, 200, 200)
    
    l_lapcount = ac.addLabel(appwindow, "Lapcount: 0")
    ac.setPosition(l_lapcount, 10, 10)
    
    l_validlap = ac.addLabel(appwindow, "invalid Lap: False")
    ac.setPosition(l_validlap, 10, 30)
    
    l_rpms = ac.addLabel(appwindow, "RPM: 0")
    ac.setPosition(l_rpms, 10, 50)

    l_speedKMH = ac.addLabel(appwindow, "Speed: 0")
    ac.setPosition(l_speedKMH, 10, 70)

    l_normalizedSplinePosition = ac.addLabel(appwindow, "Normalized Spline Position: 0")
    ac.setPosition(l_normalizedSplinePosition, 10, 90)

    l_gap = ac.addLabel(appwindow, "Gap: 0")
    ac.setPosition(l_gap, 10, 110)

    l_laptime = ac.addLabel(appwindow, "Laptime: 0")
    ac.setPosition(l_laptime, 10, 130)

    l_slipAngle = ac.addLabel(appwindow, "Slip Angle: 0")
    ac.setPosition(l_slipAngle, 10, 150)

    l_session_time_left = ac.addLabel(appwindow, "Session Time Left: 0")
    ac.setPosition(l_session_time_left, 10, 170)

    #create state
    state = State.State()

    #create thread to update state
    t = threading.Thread(target=updateSharedMem, args=(0,))
    t.start()

    ac.log("ACAI loaded")

    return "ACAI"

def acUpdate(deltaT):
    global state, i, storedLapTime, bestLapMap, currlapMap, prelap, prelaptime, lastnormalizedSplinePosition
    global l_lapcount, l_validlap, l_rpms, l_speedKMH, l_normalizedSplinePosition, l_gap, l_laptime, l_slipAngle

    # set best lap data saves between sessions
    if (info.graphics.iLastTime<storedLapTime and storedLapTime>400 and info.graphics.iLastTime>0):
        # ac.log(str(info.graphics.iBestTime))
        # currlapMap.pop(1) #remove second element of list as it is time between prelap and lap
        bestLapMap = currlapMap
        storedLapTime = info.graphics.iLastTime
        
        ac.log("new best lap time: " + str(storedLapTime))
        with open('bestLapMap.json', 'w') as f:
            f.write(json.dumps({"arr": bestLapMap,
                        "time": storedLapTime}))
            f.close()
    
    lastnormalizedSplinePosition = state.normalizedSplinePosition

    if (ac.getCarState(0, acsys.CS.LapCount)>state.lapcount):
        state.isInvalidLap = 0
        ac.log("lap finished")
        currlapMap = []
        ac.setText(l_validlap, "invalid Lap: " + str(state.isInvalidLap))

    
    state.numberOftyresOut = info.physics.numberOfTyresOut
    if (state.numberOftyresOut>2):
        state.isInvalidLap = 1
        ac.setText(l_validlap, "invalid Lap: " + str(state.isInvalidLap))

    state.rpms = info.physics.rpms

    state.lapcount = ac.getCarState(0, acsys.CS.LapCount)
    ac.setText(l_lapcount, "Lapcount: " + str(state.lapcount))

    # state.isInvalidLap = ac.getCarState(0, acsys.CS.LapInvalidated)
    # ac.setText(l_validlap, "invalid Lap: " + str(state.isInvalidLap))

    ac.setText(l_rpms, "RPM: " + str(state.rpms))

    state.speedKMH = info.physics.speedKmh
    ac.setText(l_speedKMH, "Speed: " + str(state.speedKMH))

    state.normalizedSplinePosition = info.graphics.normalizedCarPosition
    ac.setText(l_normalizedSplinePosition, "Normalized Spline Position: " + str(state.normalizedSplinePosition))

    if (round(state.normalizedSplinePosition*100) == 0 and prelap == True):
        ac.log("lap started")
        prelap = False
        currlapMap = [0]


    state.laptime = info.graphics.iCurrentTime
    ac.setText(l_laptime, "Laptime: " + str(state.laptime))


    state.slipAngle = ac.getCarState(0, acsys.CS.SlipAngle)
    ac.setText(l_slipAngle, "Slip Angle: " + str(state.slipAngle))

    state.session_time_left = info.graphics.sessionTimeLeft
    ac.setText(l_session_time_left, "Session Time Left: " + str(state.session_time_left))

    #detect session restart for 30 min hotlap session
    if (state.session_time_left>1780000 and prelap == False):
        prelap = True
        ac.log("session started")

    #lap marker updates
    if (prelap):
        lastnormalizedSplinePosition=0
        pass
    elif ((round(state.normalizedSplinePosition *1000))%5!=0):
        pass
    elif (round(lastnormalizedSplinePosition*1000) == round(state.normalizedSplinePosition* 1000)):
        pass
    elif(len(currlapMap)<2 and prelaptime == True):
        prelaptime = False
        pass
    elif (lastnormalizedSplinePosition>state.normalizedSplinePosition):
        pass
    else:
        currlapMap.append(state.laptime)
        # compare with best lap
        #get idx of current lap
        if (len(bestLapMap)>0):
            idx = len(currlapMap)-1
            state.gap = currlapMap[idx] - bestLapMap[idx]
            ac.setText(l_gap, "Gap: " + str(state.gap))
        ac.log(str(currlapMap[-1]))



def acShutdown():
    global t, stopped, shared_mem, bestLapMap, storedLapTime
    #kill thread
    stopped = True
    t.join()

    # dispose shared memory
    global shared_mem
    
    ac.log("Shared memory disposed")

    #save best lap arr and laptime
    with open('bestLapMap.json', 'w') as f:
        f.write(json.dumps({"arr": bestLapMap,
                     "time": storedLapTime}))
        f.close()
    


def setupSharedMem():
    global shared_mem
    #create a file to map in memory
    with open('acai', 'r+b') as f:
        f.write(b'\0' * 2048)
        f.flush()
        f.close()

    
    with open('acai', 'r+b') as f:
        shared_mem = mmap.mmap(f.fileno(), 2048, access=mmap.ACCESS_WRITE)
    pass

def updateSharedMem(*args):
    global state, shared_mem, stopped
    setupSharedMem()

    while not stopped:
        #update shared memory
        shared_mem.seek(0)
        shared_mem.write(state.toJSON().encode())
        shared_mem.write(b'\0' * (2048 - len(state.toJSON().encode())))
        shared_mem.flush()
        # ac.log("Shared memory updated")
        # time.sleep(0.033)
    shared_mem.close()
    shared_mem.unlink()
    return