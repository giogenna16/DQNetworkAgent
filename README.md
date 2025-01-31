# DQNetworkAgent
The goal of this project is to train agent to navigate and collect bananas! in a large, square world. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the course GitHub repository, in a folder in the same root of `Navigation.ipynb`, and unzip (or decompress) the file. 

3. Follow the instructions in `Navigation.ipynb` and install the dependencies in `requirements.txt`

4. `checkpoint.pth` contains the saved model weights of a successful agent and `score_episode_plot.pth` shows the trend of the average score over episodes during the training phase of the same agent until the goal is reached (average score of +13 over 100 consecutive episodes). Specifically, this agent reaches the gol in 425 episodes and it is based on naive DQN, with the additional use of DuelingDQN and DoubleDQN techniques.