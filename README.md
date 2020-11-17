# Reinforcement Learning

While I was reading the book *PyTorch 1.x Reinforcement Learning*, I reproduced some of the code in the book.

> Requirements:
>
> - PyTorch
> - Gym (OpenAI Gym)
> - Gym[atari] environments
> - numpy
> - matplotlib

## Basic Concept about the OpenAI Gym Environment

### Create an Environment Instance

```Python3
import gym
env = gym.make('SpaceInvaders-v0')
```

### Render an Environment

```Python3
env.render()
```

### Reset an Environment

```Python3
state = env.reset()
```

### Obtain the Number of States and Actions

```Python3
n_state = env.observation_space.n
n_action = env.action_space.n
```

### Take an Action upon a State

```Python3
new_state, reward_of_this_step, is_done, additional_info_dict = env.step(action)
```

> Note: usually the action is an integer, so is the state.

## Basic Concepts about Reinforcement Learning

### State-Value, Utility

The value of the given states.

### Action-Value, Q-Function, Q

The value of the actions in the given states.

### Policy

A strategy that tell us what action we should take when facing certain states. For example:
| State | Action |
| --- | --- |
| hungry | eat |
| tired | rest |
| waken | work |

### Discount Factor: $\gamma$

A factor to balance the further reward and current reward. If $\gamma$ is 1, then 1 future reward counts as 1 current reward. If $\gamma$ is 0.1, then future reward counts as 0.1 current reward. If $\gamma$ is 0, then future reward does not count in current state.

### Transition Probability

Usually it's a matrix of shape `n_state * n_action * n_state`, which tells us under which state, if we take which action, will result in a new state in how much probability.

### Learning Rate $\alpha$

Similar to the learning rate in neural network, $\alpha$ is controls the parameter update rate.

### Probability of Event: $\epsilon$

$\epsilon$ controls the probability of an event happening. It controls the balance between exploitation and exploration. When it approaches 1, the algorithm becomes random, which explores the environment. When it approaches 0, the algorithm becomes deterministic, which exploits the existing policy.

### Exploring Starts

Start an episode by a random step.

## Markov Decision Process (MDP)

The future state solely depends on current state and what action we will take in this environment.

### Value Iteration Algorithm

The basic idea is: calculate the value of each state in the policy, for each state, take the action which has the highest value. The value update formula is *Bellman optimal equation*: 

$V_{state} = argmax_{action}(reward_{state, action} + \gamma * \sum_{new\_state}{transition\_prob_{state, action, new\_state} * V_{new\_state}})$

Optimal policy is:

$\pi_{state} = argmax_{action}\sum_{new\_state}{transition\_prob_{state, action, new\_state} * [reward_{state, action, new\_state} + \gamma * V_{new\_state}]}$

### Policy Iteration Algorithm

Two steps are taken alternatively: policy evaluation and policy update.

Policy evaluation is through *Bellman expectation equation*:

$V_{state} = \sum_{action}\pi_{state, action}[reward_{state, action} + \gamma * \sum_{new\_state}{transition\_prob_{state, action, new\_state} * V_{new\_state}}]$

Policy update is through *Bellman optimal equaltion*:

$V_{state} = argmax_{action}(reward_{state, action} + \gamma * \sum_{new\_state}{transition\_prob_{state, action, new\_state} * V_{new\_state}})$

### How to Choose Policy Iteration or Value Iteration

- If there is a large number of actions, use policy iteration, because it converges faster.
- If there is a small number of actions, use value iteration.
- If there is already a viable policy, use policy iteration.

### Disadvantage of MDP

It requires the environment to be fully known, e.g. the transition probabilities and reward matrix, which is hard to acquire in real environment.

## Monte Carlo Methods

Based on *Law of Large Numbers (LLN)*, the average performance of a large number of repeated events or actions will eventually converge to the expected value.

Compared to *Markov Decision Process*, this is a model-free method.

### Monte Carlo Policy Evaluation

The returns of a process can be calculated as such:

$G_t = \sum_k{\gamma^k * reward_{t+k+1}}$

There are two ways to calculate the value of a state: first-visit and every-visit.

#### First-Visit

$V_{state} = \frac{1}{N_{state\_appears}} \sum_{episode}{G_{state\_first\_appears\_in\_episode}}$

#### Every-Visit

$V_{state} = \frac{1}{N_{state\_appears\_in\_how\_many\_steps}} \sum_{episode}{G_{state\_every\_time\_appears\_in\_episode}}$

### Monte Carlo Control

#### On-Policy Method

On-policy methods learn about the optimal policy by executing the policy and evaluating and improving it. It is similar to the policy iteration method in MDP.

```Python3
for each episode:
    states_playback, actions_playback, rewards_playback = run_episode(Q)
    calculate G for each state-action pair (first-visit or every-visit) based on this episode playback
    accumulate the G in G_global, using G_global to update Q
using Q to obtain the optimal policy
```

When running each episode, we can always choose the best action greedily. Alternatively, we can use epsilon-greedy policy which will choose the action based on the probability:

- best_action: $1 - \epsilon + \frac{\epsilon}{N_{action}}$
- other_action: $\frac{\epsilon}{N_{action}}$

#### Off-Policy Method

Off-policy methods learn about the optimal policy using the data generated by another policy. Importance sampling technique is used, as `W[(state, action)]`.

```Python3
behavior_policy = any policy (usually random) that can explore all the state-action pairs
for each episode:
    run an episode using behavior_policy and obtain the play_backs
    play the play_backs backwards:
        if action != best_policy under Q:
            break
        calculate G_t
        calculate w = w * 1 / behavior_policy[state][action]
        W[(state, action)] = w
    for (state, action,) return in G.items():
        G_global[(state, action)] = W[(state, action)] * return
        calculate Q[state][action] = G_global[(state, action)] / N[(state, action)]
optimal policy = argmax(Q[state]) for each state
```

Here we can use weighted importance sampling. To do that, we can just update N and Q differently:

```Python3
N[(state, action)] += w
Q[state][action] += (w / N[(state, action)]) * (return - Q[state][action])
```

### Disadvantages of Monte Carlo Methods

Some environment may be very long, so only updating the policy at the end of episode might become very inefficient.

## Q-Learning

Q-Learning is an off-policy learning algorithm which is a Temporal Difference method. The policy is updated at the end of each step, instead of episode. 

The formula is:

$Q_{state, action} = Q_{state, action} + \alpha * (reward_{state, action} + \gamma * \max_{new\_action}{Q_{new\_state, new\_action} - Q_{state, action}})$

The $\alpha * (reward_{state, action} + \gamma * \max_{new\_action}{Q_{new\_state, new\_action} - Q_{state, action}})$ part is the temporal difference.

In Q-Learning, actions are taken according to epsilon-greedy policy.

```Python3
for each episode:
    while not done:
        action = epsilon_greedy(state, Q)
        get new_state, reward info by taking action
        Temporal Difference = reward + gamma * max(Q[next_state]) - Q[state][action]
        Q[state][action] += alpha * Temporal Difference
optimal policy[state] = argmax(Q[state])
```

## Double Q-Learning

There are two Q's in Double Q-Learning: Q1 and Q2.

When we need to obtain Q, we just add Q1 and Q2: `Q = Q1 + Q2`.

The difference from Q-Learning is: 

1. Find the action from (state, Q1+Q2)
2. After taking the action, randomly update Q1 or Q2 based on the following formula.

If Q1 is selected to update:

$a^* = \argmax_a{Q1_{new\_state, action}}$

$Q1_{state, action} = Q1_{state, action} + \alpha*(reward + \gamma*Q2_{new\_state, a^*} - Q1_{state, action})$

If Q2 is selected to update:

$a^* = \argmax_a{Q2_{new\_state, action}}$

$Q2_{state, action} = Q2_{state, action} + \alpha*(reward + \gamma*Q1_{new\_state, a^*} - Q2_{state, action})$

## State-Action-Reward-State-Action (SARSA) Algorithm

SARSA is an on-policy Temporal Difference algorithm.

The formula is:

$Q_{state, action} = Q_{state, action} + \alpha * (reward_{state, action} + \gamma * Q_{new\_state, new\_action} - Q_{state, action})$

```Python3
for each episode:
    action = epsilon_greedy(state, Q)
    while not done:
        get new_state, reward info by taking action
        Temporal Difference = reward + gamma * Q[next_state][next_action] - Q[state][action]
        Q[state][action] += alpha * Temporal Difference
        action = next_action
optimal policy[state] = argmax(Q[state])
```

The differences from Q-Learning are:

1. Q\[next_state]\[next_action] instead of max(Q\[next_state])
2. action = next_action which is why it's called on-policy algorithm.

## (Contextual) Multi-Armed Bandit

In this model, we can take several actions, and every action corresponds to certain reward, we need to decide which action to take based on the policy. Usually it does not involve context (state) information. If we consider context (state) in Multi-Armed Bandit problem, every Multi-Armed Bandit machine has its own payout probabilities and reward, which is the "context" or "state" when talking in multi-machine context.

The biggest difference from MDP is that in Multi-Armed Bandit problem, the action does not depend on the previous state, and that in Contextual (multi-machine) Multi-Armed Bandit problem, the action does not depend on the previous state either.

To develop a reinforcement learning algorithm for Multi-Armed Bandit problem, we can use a policy function to obtain the action based on Q or action appearing count.

### Epsilon Greedy Policy

It will choose the action based on the probability:

- best_action: $1 - \epsilon + \frac{\epsilon}{N_{action}}$
- other_action: $\frac{\epsilon}{N_{action}}$

### Softmax Exploration Policy

It will not choose the non-best actions based on a fixed equal probabilities, instead, it will choose the action based on the softmax of the Q function.

$prob = softmax(Q)$

### Upper Confidence Bound

As the training proceeds, the interval that the action value falls into becomes narrower. We can use the upper bound of this interval to generate the action. The algorithm is greedy.

$action = \argmax(Q + \sqrt{2*\log(episode)/action\_count})$

where action_count is the list recording how many times each action appears till this episode.

This algorithm is different from Epsilon Greedy and Softmax Exploration in that it will change probabilities distribution over time.

### Thompson Sampling Policy

This algorithm will use [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution) to generate the action.

$action = \argmax{\Beta(\alpha, \beta)}$

where $\alpha$ is how many times the reward appears for each action, and $\beta$ is how many times the reward does not appear for each action.

### Contextual Multi-Armed Bandit Problem

The difference between Contextual and Non-Contextual Multi-Armed Bandit problems is that we need to maintain the Q for each state (or machine, in the context of Multi-Armed Bandit machines).

## Function Approximation

When the number of states is countless, we cannot use a lookup table to obtain the next action, in which case, we should turn to Function Approximation.

Function Approximation is a method to estimate the Q values of a state using a function (maybe linear, maybe nonlinear, e.g., Neural Network). The function can accept the a parameter, state, and return the Q values for the given state, Q, which is a vector whose length is the number of actions.

## Deep Q-Networks (DQN)

Using a neural network to predict Q-values for all actions on the state. The network can be a fully connected networks, or a convolutional neural networks (CNNs).

### Experience Replay

To improve the efficiency of learning, we can accumulate the `(state, action, reward, next_state, is_done)` in the memory. When the memory is long enough, a batch of random samples can be drawn from the memory and used for training the model.

### Double Deep Q-Networks

To predict the Q-values of `next_state`, we can use a separate network (called `target model`) to predict the Q-values of `next_state` and update it once per several episodes. By updating it, we mean copying the weights of the learning model to the `target model`.

### Dueling Deep Q-Networks (DDQN)

We can decouple the Deep Q-Networks into two parts:

- `V(state)`: state-value function, calculating the value of being at the state
- `A(state, action)`: state-dependent action advantage function, estimating how much better it is to take the action, rather than taking other actions at the state.

By decoupling the value and advantage functions, we are able to accommodate the fact that our agent may not necessarily look at both the value and the advantage at the same time during the learning process. In other words, the agent using DDQNs can efficiently optimize either or both functions as it prefers.

The Q-values can be calculated as such: $Q(state, action) = V(state) + A(state, action) - \frac{1}{N_{action}}\sum_a{A(state, action)}$.

## Reference

Book: *PyTorch 1.x Reinforcement Learning*
