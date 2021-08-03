from typing import Any

import gym

from .definitions import Pose, Velocity
from .vehicles import Vehicle
from racecar_gym.tasks import Task


class Agent:

    def __init__(self, id: str, vehicle: Vehicle, task: Task, flatten: bool = False):
        self._id = id
        self._vehicle = vehicle
        self._task = task
        self._flatten = flatten

    @property
    def id(self) -> str:
        return self._id

    @property
    def vehicle_id(self) -> Any:
        return self._vehicle.id

    @property
    def action_space(self) -> gym.Space:
        if self._flatten:
            return gym.spaces.utils.flatten_space(self._vehicle.action_space)
        else:
            return self._vehicle.action_space

    @property
    def observation_space(self) -> gym.Space:
        if self._flatten:
            return gym.spaces.utils.flatten_space(self._vehicle.observation_space)
        else:
            return self._vehicle.observation_space

    def action_unflatten(self, action):
        if self._flatten:
            action = gym.spaces.utils.unflatten(self._vehicle.action_space, action)

    def observation_flatten(self, observation):
        if self._flatten:
            action = gym.spaces.utils.flatten(self._vehicle.action_space, observation)

    def step(self, action):
        observation = self._vehicle.observe()
        self._vehicle.control(action)
        return observation, {}

    def done(self, state) -> bool:
        return self._task.done(agent_id=self._id, state=state)

    def reward(self, state, action) -> float:
        return self._task.reward(agent_id=self._id, state=state, action=action)

    def reset(self, pose: Pose):
        self._vehicle.reset(pose=pose)
        self._task.reset()
        observation = self._vehicle.observe()
        return observation
