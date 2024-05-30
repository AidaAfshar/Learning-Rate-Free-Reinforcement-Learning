# Learning-Rate-Free-Reinforcement-Learning
This repository includes the Learning rate-free version of PPO and DQN, based on model selection. The paper preprint will be available  after double-blind review.

For choosing the model selection strategy, simply set the 'modsel_alg' parameter to one of the following during the initialization:
- "BHD3"
  - for Doubling Data-Driven Regret Balancing algorithm
- "Classic"
  - for the regret bound balancing algorithm
- "Corral"
- "UCB"
- "Exp3"


## Citations
For Model Selection methods, we use the implementations available at [model selection](https://github.com/pacchiano/modelselection) repository.

For reinforcement learning algorithms, we used the [cleanRL](https://docs.cleanrl.dev/) library. 
