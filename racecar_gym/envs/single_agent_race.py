from typing import Dict, Any
import gym
from .scenarios import SingleAgentScenario

class SingleAgentRaceEnv(gym.Env):

    metadata = {'render.modes': ['follow', 'birds_eye', 'lidar']}

    def __init__(self, scenario: SingleAgentScenario):
        self._scenario = scenario
        self._initialized = False
        self._time = 0.0
        self.observation_space = scenario.agent.observation_space
        self.action_space = scenario.agent.action_space

    @property
    def scenario(self):
        return self._scenario        

    def _step_dict(self, action: Dict):
        assert self._initialized, 'Reset before calling step'
        state = self._scenario.world.state()
        observation, info = self._scenario.agent.step(action=action)
        observation['time'] = self._time
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        self._time = self._scenario.world.update()
        return observation, reward, done, state[self._scenario.agent.id]

    def _step_array(self, action: Any):
        assert self._initialized, 'Reset before calling step'
        state = self._scenario.world.state()
        action = self._scenario.agent.action_unflatten(action)
        observation, info = self._scenario.agent.step(action=action)
        # observation['time'] = self._time
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        self._time = self._scenario.world.update()
        observation = self._scenario.agent.observation_flatten(observation)
        return observation, reward, done, state[self._scenario.agent.id]

    def step(self, action: Any):
        if self._scenario.agent._flatten:
            self._step_array(action)
        else:
            self._step_dict(action)

    def reset(self, mode: str = 'grid'):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        obs = self._scenario.agent.reset(self._scenario.world.get_starting_position(self._scenario.agent, mode))
        self._scenario.world.update()
        if self._scenario.agent._flatten:
            obs = self._scenario.agent.observation_flatten(obs)
        else:
            obs['time'] = 0
        return obs

    def render(self, mode: str = 'follow', **kwargs):
        return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **kwargs)

    def seed(self, seed=None):
        self._scenario.world.seed(seed)