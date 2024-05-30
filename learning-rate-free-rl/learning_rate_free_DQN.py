import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from model_selection_algorithms.algorithmsmodsel import BalancingHyperparamDoublingDataDriven, CorralHyperparam, \
    EXP3Hyperparam, UCBHyperparam, BalancingClassic
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    # Model Selection Algorithm Parameters
    modsel_options = ["BHD3", "Corral", "UCB", "Exp3", "Classic"]
    modsel_alg: str = "Classic"
    hparam_to_tune: str = "learning_rate"
    num_base_learners: int = 10
    base_learners_hparam = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    #num_base_learners: int = 1
    #base_learners_hparam = [2.5e-4]
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = f"Modsel_DQN_lr"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    # Algorithm specific arguments
    env_id: str = "Acrobot-v1"
    """the id of the environment"""
    total_timesteps: int = 5*500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def normalize_episodic_return(episodic_return, normalizer_const, args):
    if args.env_id == "Acrobot-v1":
        normalized_return = (episodic_return + 500) / 500
    else:
        if episodic_return < normalizer_const:
            normalized_return = episodic_return/normalizer_const
        else:
            normalized_return = 1
    return normalized_return

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
# ALGO LOGIC: initialize agent here:


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )
    def forward(self, x):
        return self.network(x)
class BaseLearner(nn.Module):
    def __init__(self, base_index, envs, device, args):
        super().__init__()
        self.base_index = base_index
        self.q_network = QNetwork(envs).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.base_learners_hparam[base_index])
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
    def get_base_learner_components(self):
        return self.q_network, self.target_network, self.optimizer, self.rb
    def update_target_network(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
            )
if __name__ == "__main__":
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"DQN__{args.modsel_alg}__lr__{args.env_id}__{args.seed}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # base learners initiation
    m = args.num_base_learners
    base_learners = []
    for i in range(m):
        agent = BaseLearner(base_index=i, envs=envs, device=device, args=args).to(device)
        base_learners.append(agent)
    selected_base_learners = []
    # meta learner initiation
    if args.modsel_alg == "BHD3":
        modsel = BalancingHyperparamDoublingDataDriven(m, dmin=1)
    elif args.modsel_alg == "Corral":
        modsel = CorralHyperparam(m)
    elif args.modsel_alg == "Exp3":
        modsel = EXP3Hyperparam(m)
    elif args.modsel_alg == "UCB":
        modsel = UCBHyperparam(m)
    elif args.modsel_alg == "Classic":
        putative_bounds_multipliers = [1] * m
        modsel = BalancingClassic(m, putative_bounds_multipliers)
    modsel_flag = True  # determines whether we should sample new base learner
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if modsel_flag:
            base_index = modsel.sample_base_index()
            selected_base_learners.append(base_index)
            q_network, target_network, optimizer, rb = base_learners[base_index].get_base_learner_components()
            modsel_flag = False
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    episodic_return = info['episode']['r']
                    normalized_return = normalize_episodic_return(episodic_return, normalizer_const=500, args=args)
                    modsel.update_distribution(base_index, normalized_return)
                    writer.add_scalar("modelselection/metalearner_normalized_episodic_return", normalized_return,
                                      global_step)
                    writer.add_scalar(f"modelselection/baselearner_{base_index}_episodic_return", normalized_return,
                                      global_step)
                    modsel_flag = True
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("modelselection/learning_rate", args.base_learners_hparam[base_index],
                                      global_step)
                    writer.add_scalar("modelselection/selected_base_learner", base_index, global_step)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # update target network
            if global_step % args.target_network_frequency == 0:
                for i in range(m):
                    base_learners[i].update_target_network()
    print("-------------------------------------")
    if args.modsel_alg != "Exp3":
        writer.add_text('base_probs', f'p={modsel.base_probas}', global_step)
        print(f'base_probas = {modsel.base_probas}')
    if args.modsel_alg == "BHD3" or args.modsel_alg == "Classic":
        writer.add_text('num_plays', f'num_plays={modsel.num_plays}', global_step)
        print(f'num_plays={modsel.num_plays}')
    if args.modsel_alg == "BHD3":
        writer.add_text('balancing_potentials', f'phi={modsel.balancing_potentials}', global_step)
        print(f'phi={modsel.balancing_potentials}')
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()
