# ACAI

RL based racing implemetation fo Assetto Corsa

does not use CV instead uses all info taken from sim data

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

- inside the directory "ac-app" copy the ACAI folder and paste into <Assetto Corsa install dir>\apps\python

- in content manager enable content>miscellaneous ACAI should be present

    - ensure that it is activated

- inside AC_req_files copy all files and paste into Assetto Corsa install folder
    - these files are essential as these are the maps for track limits and the shared memory file for state updates
    - track limit files are specific to daytona, new tracks will need to have different maps. I have code to make this happen but its very manual so if there is a request i will make instructions.


- changes should now be done inside the assetto corsa directory, updates require either a live reload in the python debugger app in assetto corsa (only available when custom shader patch installed) or restart session entirely (close window and restart session via Content manager)

#### setup ACAI

- in the root directory of this repo:
```
pip3.10 install -e .

```
- adjust file paths in ACAI\ACAI\envs\ACAI_AI.py to match what is installed

- edits in this directory will be live updated, no need to rerun install

##### Register ACAI to rl_zoo3
- navigate to C:\Users\\<User>\AppData\Roaming\Python\Python310\site-packages\rl_zoo3
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

...

```

### Training and hyperparams

- in directory C:\Users\\<User>\AppData\Roaming\Python\Python310\site-packages\rl_zoo3\hyperparams there are hyperparameters for each implemented algorithm. 
    - the environment State is a Dict and outputs are continuous therefore only some algs will work

- in this implementation td3.yaml was updated with the following lines:

```
...

ACAI-v0:
  normalize: True
  n_envs: 1
  n_timesteps: !!float 1e6
  policy: 'MultiInputPolicy'
  learning_rate: !!float 7.3e-4
  buffer_size: 300000
  batch_size: 256
  # ent_coef: 'auto'
  gamma: 0.99
  tau: 0.02
  train_freq: 8
  gradient_steps: 10
  learning_starts: 0
  # use_sde: True
  # use_sde_at_warmup: True
```
- in this implementation sac.yaml was updated with the following lines:

```
...

ACAI-v0:
  # frame_stack: 4
  normalize: True
  n_envs: 1
  n_timesteps: !!float 1e6
  policy: 'MultiInputPolicy'
  learning_rate: !!float 7.3e-4
  buffer_size: 300000
  batch_size: 256
  ent_coef: 'auto'
  gamma: 0.99
  tau: 0.02
  train_freq: 8
  gradient_steps: 10
  learning_starts: 1000
  use_sde: True
  use_sde_at_warmup: True
```
#### Start Training

- open content manager select car and track
    - daytona 2017 oval (found on racedepartment)
    - car can be anything, only trained on Lotus Exos 125

- set conditions and ensure you are using hotlap mode with timelimit to 30 minutes (timelimit should be automatically to 30 mins)
    - essential because some detection code relies on session timer
    - enable damage (200% ensures if wall is hit at any speed damage will happen therfore reset)
    - turn on automatic shifting and clutch, this is because the environment wrapper only outputs steering between -1 to 1 and throttle (brake was not implemented as it is not a fast strategy on daytona, can be added but greatly increases training time)
    - enable traction and stability control, prevents lighting up rears on start.

- start session 
    - training create a logs folder outside of this repo directory and in that folder run:

```
python3.10 -m rl_zoo3.train --algo td3 --env ACAI-v0 --save-freq 10000
```
- Ctrl-C can work safely 


### Run the trained model

- anywhere run:
```
python3.10 -m rl_zoo3.enjoy --algo td3 --env ACAI-v0 -f <path-to-your-log-folder>\logs --exp-id <your_exp_num> --load-checkpoint <iters> --norm-reward --no-render
```

- logs folder struct
```
<path-to-your-log-folder>\logs
|--sac
|--|ACAI-v0_1
|--|--ACAI-v0
|--|--|args.yml
|--|--|config.yml
|--|--|vecnormalize.pkl
|--|--0.monitor.csv
|--|--best_model.zip
|--|--evaluations.npz
|--|--rl_model_10000_steps.zip
|--|--rl_model_20000_steps.zip
|--|--rl_model_<iters>_steps.zip
|
|--|ACAI-v0_2
|--|ACAI-v0_<your_exp_num>
|--td3
|--|ACAI-v0_1
|--|ACAI-v0_2
|--|ACAI-v0_...
```

- each time you run the train command it creates a new ACAI-v0_<your_exp_num>
- every 10000 \<iters> it saves a model

# overview information
- presentation can be found here: https://youtu.be/hJRGfN3b2EE
- create issue report for questions or contact via kalla100@umn.edu or archkalla@gmail.com (cc both please)













