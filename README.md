# Learning-Rate-Free-Reinforcement-Learning
This repository includes the Learning rate-free version of PPO and DQN, using model selection algorithms. 

For choosing the model selection strategy, set the 'modsel_alg' parameter to one of the following during the initialization:
- "D3RB"
  - for Doubling Data-Driven Regret Balancing algorithm 
- "ED2RB"
  - for Estimating Data-Driven Regret Balancing   
- "Classic"
  - for the regret bound balancing algorithm
- "Corral"
- "UCB"
- "Exp3"


## Citations
Model Selection implementations are originally from [model selection](https://github.com/pacchiano/modelselection) repository.

Reinforcement learning algorithms are modified versions of [cleanRL](https://docs.cleanrl.dev/) implementations. 
