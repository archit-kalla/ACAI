import State
import mmap
with open("C:\Program Files (x86)\Steam\steamapps\common\\assettocorsa\\acai", 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
state = State.State()
def get_state_shared_mem():
    global mm, state
    #read shared memory
    mm.seek(0)
    data = mm.read(2048).decode('utf-8')
    # remove bytes after '\0'
    data = data[:data.find('\0')]

    #replace all \0 with spaces
    data = data.replace('\0', '')
    # print(data)
    state.from_json(data)

    print(state.isInvalidLap, end='\r')

   
    

while True:
    get_state_shared_mem()