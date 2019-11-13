import numpy as np
import torch
from torch.optim import Adam
from itertools import chain

import gym

class Trainer:

    def __init__(self, env, agent, lr):
        self.env = env
        self.agent = agent
        self.optimizer = Adam(chain(self.agent.actor.parameters(), self.agent.critic.parameters()), lr=lr)
        self.init_memory()

    def init_memory(self):
        self.memory = {
                    "log_probs" : [],
                    "values" : [],
                    "rewards" : [],
                    "masks" : [],
                }

    def test(self, render=False):
        env = gym.make("CartPole-v0")
        state = env.reset()
        done = False
        ret = 0.0
        while not done:
            if render:
                env.render()
            action = self.agent.choose_action(state).numpy()
            next_state, reward, done, _= env.step(action)
            ret += reward
            state = next_state
        env.close()
        print(ret)
        return ret


    def append_to_memory(self, log_prob, reward, value, mask):
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(reward)
        self.memory["values"].append(value)
        self.memory["masks"].append(mask)


    def train(self, epochs, max_steps, test_every=None):
        current_state = self.env.reset()
        for epoch in range(epochs):
            self.init_memory()
            entropy = 0
            for _ in range(max_steps):
                dist = self.agent.get_dist(current_state)
                action = dist.sample()

                next_state, reward, done, _ = self.env.step(action.numpy())

                entropy += dist.entropy().mean()
                log_prob = dist.log_prob(action)

                value = self.agent.get_value(current_state)
                self.append_to_memory(log_prob, reward, value, (1 - done))

                current_state = next_state
                frame += 1

            if test_every is not None and epoch % test_every == 0:
                self.test()

            next_value = self.agent.get_value(torch.FloatTensor(next_state))

            rewards = torch.FloatTensor(np.concatenate(self.memory["rewards"]))
            masks = torch.FloatTensor(np.concatenate(self.memory["masks"]))
            returns = torch.stack(Trainer.compute_returns(next_value, rewards, masks))

            log_probs = torch.cat(self.memory['log_probs'])
            values = torch.cat(self.memory['values']).squeeze()

            advantages = returns - values

            actor_loss  = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    @staticmethod
    def compute_returns(last_value, rewards, masks, gamma=0.99):
        R = last_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns