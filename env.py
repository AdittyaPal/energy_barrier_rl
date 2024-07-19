import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MolecularPath(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to find a pathway.
    """

    def __init__(self, initial_pos, final_pos, max_steps = 1000, num_dimens = 2, scale = 0.09):
        super(MolecularPath, self).__init__()
        
        # Size of the nD-grid
        self.scale = scale
        self.num_dimens = num_dimens
        # Initialize the agent at some arbitrary position
        self.agent_pos = initial_pos
        self.agent_start = initial_pos
        self.agent_stop = final_pos
        
        self.max_steps = max_steps
        self.steps = 0

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low = np.array([-1.0, -1.0]), high = np.array([1.0, 1.0]), shape=(num_dimens,), dtype = np.float32)
        #self.action_space = spaces.Box(low = -1, high = 1, shape=(num_dimens,), dtype = np.float32)
        # The observation will be the coordinates of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low = np.array([-5.0, -2.0]), high = np.array([5.0, 8.0]), 
                                            shape=(num_dimens,), dtype = np.float32)

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        self.agent_pos = np.array(self.agent_start) + np.random.normal(scale = 0.1, size = 2)
        self.steps = 0
        # here we convert to int32 because we are using disrete actions
        return np.array([self.agent_pos]).astype(np.float32).reshape(self.num_dimens,), {}  # empty info dict
    
    def step(self, action, states):
        truncated = False
        terminated = False
        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        self.steps += 1
        copy_pos = self.agent_pos
        self.agent_pos = self.agent_pos + self.scale * action
        for i in range(self.num_dimens):
            self.agent_pos[i] = np.clip(self.agent_pos[i], 
            				self.observation_space.low[i], self.observation_space.high[i])
        
        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = -self.get_reward(self.agent_pos)
        
        if (self.steps > self.max_steps):
            truncated = True
        # Has the agent reached the target?
        if (np.sum(np.square(self.agent_pos - np.array([-0.558, 1.442])) < 1e-8)):
            terminated = True
            #reward += np.sum(np.logspace(start = 0, stop = self.max_steps - self.steps, num = self.max_steps - self.steps, base = 0.99)) * reward
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.agent_pos]).astype(np.float32).reshape(self.num_dimens,),
            reward,
            terminated,
            truncated,
            info,
        )
    
    def action(self, sample_action):
    	return np.array(sample_action)
    
    def get_reward(self, pos):
        # agent is represented as a cross, rest as a dot
        x = pos[0]
        y = pos[1]
        
        t1 = -200 * np.exp((-1.0)*(x - 1.0)**2 +                            + (-10.0)*(y      )**2)
        t2 = -100 * np.exp((-1.0)*(x      )**2 +                            + (-10.0)*(y - 0.5)**2)
        t3 = -170 * np.exp((-6.5)*(x + 0.5)**2 + (11.0)*(x + 0.5)*(y - 1.5) + (- 6.5)*(y - 1.5)**2)
        t4 =   15 * np.exp(( 0.7)*(x + 1.0)**2 + ( 0.6)*(x + 1.0)*(y - 1.0) + (  0.7)*(y - 1.0)**2)
        
        '''
        d_t1_x = - 2.0 * (x - 1.0) * t1
        d_t1_y = -20.0 * (y      ) * t1
        
        d_t2_x = - 2.0 * (x      ) * t2
        d_t2_y = -20.0 * (y - 0.5) * t2
        
        d_t3_x = (-13.0 * (x + 0.5) + 11.0 * (y - 1.5)) * t3
        d_t3_y = ( 11.0 * (x + 0.5) - 13.0 * (y - 1.5)) * t3
        
        d_t4_x = (  1.4 * (x + 1.0) +  0.6 * (y - 1.0)) * t4
        d_t4_y = (  0.6 * (x + 1.0) +  1.4 * (y - 1.0)) * t4        
        '''
        return t1 + t2 + t3 + t4
        #	+ 0.2 * np.sqrt((d_t1_x + d_t2_x + d_t3_x + d_t4_x)**2 
        # 			+ (d_t1_y + d_t2_y + d_t3_y + d_t4_y)**2))

    
    def close(self):
        pass


