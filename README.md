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







