# ACAI

RL based racing implemetation fo Assetto Corsa

## SETUP INSTRCTION

- Ensure you have python 3.3 installed for Assetto Corsa Environment
- Ensure you have python 3.10 installed for AI packages.
- Requires stablebaselines3 and RL zoo 3
- install vjoy, ensure that virutal device 1 is enabled.

### Recommended installation method
- use chocolatey to install python 3.10 
- use standard installer to install python 3.3 

This ensures that python directories are at the C:\ level

### Install depedencies


#### Python 3.3
- none uses default packages
#### Python 3.10


```
pip3.10 install torch torchvision torchaudio

pip3.10 install gymnasium --user

pip3.10 install pyvjoy --user

pip3.10 install stable-baselines3[extra] --user

pip3.10 install rl_zoo3 --user
```

#### SETUP ASSETTO CORSA

- ensure content manager is installed

- enable Custom Shader patch for debugging (enables python debugger)

    - in long training sessions removing the patch may be necessary to improve stability

- place "ai_controller_input_settings.ini" in ~Documents\Assetto Corsa\cfg\controllers\savedsetups

- load these setting in content manager

#### setup ACAI

- in the root directory:
```
pip3.10 install -e .

```
##### Register ACAI to rl_zoo3
- navigate to C:\Users\<User>\AppData\Roaming\Python\Python310\site-packages\rl_zoo3
- edit import_envs.py
- add the following lines of code to the try catch series of statements
```
...
try:
    import ACAI
    # gym.make("ACAI-v0")
except ImportError:
    print("ACAI not installed")
    pass
 ```







