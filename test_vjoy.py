import pyvjoy
import time

vj = pyvjoy.VJoyDevice(1)

# convert -1.0 to 1.0 to 0x1 to 0x8000
def convert_axis(value):
    scaled_value = int((value + 1) * 16384)

        # Convert the scaled integer to hexadecimal
    hex_representation = hex(scaled_value)
    print(hex_representation)
    return scaled_value

vj.set_button(2,1)
time.sleep(1)
vj.set_button(2,0)

vj.set_axis(pyvjoy.HID_USAGE_Z, convert_axis(0.0))
time.sleep(1)
vj.set_axis(pyvjoy.HID_USAGE_X, convert_axis(0.5))
#cycle 0.0 to 0.5 to 1.0 to 0.5 to 0.0
while True:

    vj.data.wAxisZ = 0x1
    vj.update()
    time.sleep(1)
    vj.data.wAxisZ = 0x4000
    
    vj.update()
    time.sleep(1)
    

#when left is pressed down, the car turns left
