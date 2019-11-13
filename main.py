from trainer import Trainer
from agent import Agent
from multiprocessing_env import SubprocVecEnv
import gym

def main():
    num_envs = 16
    env_name = "CartPole-v0"

    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    env = gym.make("CartPole-v0")

    STATE_SIZE  = env.observation_space.shape[0]
    N_ACTIONS = env.action_space.n

    agent = Agent(STATE_SIZE, N_ACTIONS)

    trainer = Trainer(envs, agent, lr=3e-4)
    trainer.train(epochs=10000, max_steps=5, test_every=50)
        # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()