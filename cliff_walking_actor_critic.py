import torch
import gym
from policy_network import ActorCriticPolicyNetwork
from utils import draw


def actor_critic(env, estimator, n_episode, gamma=1.0):
    """Actor-critic algorithm

    Args:
        env: OpenAI gym environment
        estimator: policy network
        n_episode: number of episodes
        gamma: the discount factor
    Returns:
        total reward for each episode (a list)
    """
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        state = env.reset()

        while True:
            one_hot_state = [0] * 48
            one_hot_state[state] = 1
            action, log_prob, state_value = estimator.get_action(one_hot_state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)

            if is_done:
                returns = []
                Gt = 0
                power = 0
                for reward in reversed(rewards):
                    Gt += gamma ** power * reward
                    power += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs, state_values)
                print(f"Episode {episode}, total reward {total_reward_episode[episode]}")
                if total_reward_episode[episode] >= -14:
                    estimator.scheduler.step()
                break
            
            state = next_state
    
    return total_reward_episode    


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')

    n_state = 48
    n_action = env.action_space.n
    n_hidden = [128, 32]
    lr = 0.03

    policy_net = ActorCriticPolicyNetwork(n_state, n_action, n_hidden, lr)

    gamma = 0.9
    n_episode = 1000
    
    rewards = actor_critic(env, policy_net, n_episode, gamma)

    draw("cliff_walking_actor_critic.png", "Rewards over time", "episode", "reward", rewards)