import torch
from model import Actor, Critic

class Agent:
    '''Agent is responsible for taking
    actions given an environment state
    '''
    def __init__(self, observation_size, n_actions, actor_model=None, critic_model=None):
        '''
        # Arguments:
            observation_size: length of array representing state
            n_actions: number of discrete actions available to the agent
            actor_model: (optional) A PyTorch module that
                         takes in a state as input and 
                         returns a distribution
            critic_model: (optional) A PyTorch module that
                          takes in a state as input and returns a
                          scalar value denoting the value of the state
        '''
        self.observation_size = observation_size
        self.n_actions = n_actions

        self.actor = actor_model
        if self.actor is None:
            self.actor = Actor(observation_size, n_actions)

        self.critic = critic_model
        if self.critic is None:
            self.critic = Critic(observation_size)

    def get_dist(self, state):
        '''
        # Arguments:
            state: A numpy array representing the state
                    of the environment
        # Returns:
            A torch.distribution from which actions
            can be sampled
        '''
        state = torch.FloatTensor(state)

        return self.actor(state)

    def choose_action(self, state):
        '''Picks an action given a state

        # Arguments:
            state: A numpy array representing the state
                    of the environment
        
        # Returns:
            A single value denoting the action
            to be taken
        '''
        return self.get_dist(state).sample()
    
    def get_value(self, state):
        '''Returns the estimated value of
        the given state

        # Arguments:
            state: A numpy array representing the state
                    of the environment
        
        # Returns:
            A scalar denoting the value of given state
        '''
        state = torch.FloatTensor(state)
        return self.critic(state)