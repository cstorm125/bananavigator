# Bananavigator
Q-learning Agent to Collect Bananas in a Unity ML-Toolkit Environment

Bananavigator is a Q-learning agent trained to play [Unity ML-Agent Toolkit](https://github.com/Unity-Technologies/ml-agents)'s [Banan Collector](https://www.youtube.com/watch?v=heVMs3t9qSk). We train a single agent to collect the most `yellow` bananas (+1) as possible while avoiding `blue` bananas (-1). We augmented [vanilla DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) with [DDQN](https://arxiv.org/abs/1509.06461), [Dueling Networks](https://arxiv.org/abs/1511.06581), [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), and [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295). The model solved the environment (scoring a 100-play moving average of 13 or above) in under 200 episodes (15 minutes on CPU), 9 times faster than the [benchmark implementation](https://classroom.udacity.com/nanodegrees/nd893/parts/6b0c03a7-6667-4fcf-a9ed-dd41a2f76485/modules/4eeb16ab-5ac5-47bf-974d-12784e9730d7/lessons/69bd42c6-b70e-4866-9764-9bfa8c03cdea/concepts/0df81599-d934-4dfb-860c-22f723129795). The weights of trained Q-network are saved as `half_rainbow.m`.

## Environment

The environment consists of 37 values for a state and 4 available actions. The state represents the agent's velocity and ray-based perception of objects around the agent's forward direction. The actions are move forward, move backward, turn left and turn right respectively. The environment will run for 300 transitions. When the agent runs into a `yellow` banana, it gets +1 reward and when it runs into a `blue` banana, it gets -1 reward. These rewards sum up to a score at the end of each episode. The environment is considered solved when the average score of the last 100 episodes exceed 13.

```
Actions look like: 
* 0 - forward
* 1 - backward
* 2 - left
* 3 - rigth
Action size: 4
States look like: [ 1.          0.          0.          0.          0.84408134  0.          0.
  1.          0.          0.0748472   0.          1.          0.          0.
  0.25755     1.          0.          0.          0.          0.74177343
  0.          1.          0.          0.          0.25854847  0.          0.
  1.          0.          0.09355672  0.          1.          0.          0.
  0.31969345  0.          0.        ]
State size: 37
```

## Getting Started

1. Clone this repository and install `unityagents`.

```
pip -q install ./python
```

2. Import `UnityEnvironment` and load Banana Collector.

```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
```

3. Follow `train_agent.ipynb` to train the agent.

4. Our implementation is divided as follows:
* `replay_memory.py` - Experience Replay Memory
* `agent.py` - Agent
* `qnetwork` - Q-networks for local and target

## Train Agent

These are the steps you can take to train the agent with default settings.

1. Create a experience replay memory.

```
#prioritized memory
mem = PrioritizedMemory(capacity = 3000)
```

2. Create an agent.

```
a = Agent(state_size = 37, action_size = 4, replay_memory = mem, seed = 1412,
          lr = 1e-3 / 4, bs = 64, nb_hidden = 128,
          gamma=0.99, tau= 1/nb_transitions, update_interval = 5) 
```

3. Train the agent.

```
scores = []
moving_scores = []
moving_nb = 100

start_time = datetime.now()
for i in trange(500):
    env_info = env.reset(train_mode=True)[brain_name] 
    state = env_info.vector_observations[0]            
    score = 0
    #300 time steps per episode
    timestep = 0
    while True:
        #select action
        action = a.act(state,i)  

        #step
        env_info = env.step(action)[brain_name]        
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]                  

        a.step(state,action,reward,next_state,done,i)
          
        score += reward                                
        state = next_state                          
        if done: break
            
    #append scores
    scores.append(score)
    if i > moving_nb:
        moving_score = np.mean(scores[i-moving_nb:i])
        moving_scores.append(moving_score)
    else:
        moving_scores.append(0)

    if i % 101 == 0: 
#         print(f'Play {i}: {datetime.now() - start_time} Moving average: {moving_scores[-1]}')
        plt.clf()
        plt.plot(scores)
        plt.plot(moving_scores)
        
    #solved at 13
    if moving_scores[-1] > 13: 
        print(f'Solved at Play {i}: {datetime.now() - start_time} Moving average: {moving_scores[-1]}')
        break
```