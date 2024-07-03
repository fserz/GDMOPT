import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np

# AIGCEnv 继承自 gym.Env，它是自定义 Gym 环境的基础类。
class AIGCEnv(gym.Env):

    # 初始化环境的各个属性：
    def __init__(self):

        self._flag = 0
        # Define observation space based on the shape of the state
        # 定义观察空间的维度和取值范围，这里基于状态的形状定义为 0 到 1 之间的连续值空间。
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        # Define action space - discrete space with 3 possible actions
        # 定义动作空间，这里为 0 到 9 之间的离散值空间（即 10 个可能的动作）。
        self._action_space = Discrete(2*5)
        # 记录当前回合的步数，初始化为 0。c
        self._num_steps = 0
        # 记录当前回合的步数，初始化为 0。
        self._terminated = False
        # 分别记录上一个状态和最后一次专家动作。
        self._laststate = None
        self.last_expert_action = None
        # Define the number of steps per episode
        # 定义每个回合的最大步数，这里为 1。
        self._steps_per_episode = 1

    # 返回定义好的观察空间
    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    # 返回定义好的动作空间
    @property
    def action_space(self):
        # Return the action space
        return self._action_space

    # 从均匀分布中生成 1 到 2 之间的 5 个随机数。
    @property
    def state(self):
        # Provide the current state to the agent
        # rng = np.random.default_rng(seed=0)
        # states1 = rng.uniform(1, 2, 5) 从均匀分布中生成 1 到 2 之间的 5 个随机数。
        # states2 = rng.uniform(0, 1, 5) 从均匀分布中生成 0 到 1 之间的 5 个随机数。

        states1 = np.random.uniform(1, 2, 5)
        states2 = np.random.uniform(0, 1, 5)

        # 初始化为 0
        reward_in = []
        reward_in.append(0)
        # 将 states1、states2 和 reward_in 连接成一个状态向量。
        states = np.concatenate([states1, states2, reward_in])

        # 将 states1 和 states2 连接成信道增益向量。
        self.channel_gains = np.concatenate([states1, states2])
        # 更新为当前状态。
        self._laststate = states
        return states


    # 执行动作并返回环境的下一个状态、奖励、回合是否终止和额外信息
    def step(self, action):
        # Check if episode has ended
        # 检查回合是否已经终止。
        assert not self._terminated, "One episodic has terminated"
        # Calculate reward based on last state and action taken
        # 调用 CompUtility 计算奖励、专家动作、子专家动作和实际动作。
        reward, expert_action, sub_expert_action, real_action = CompUtility(self.channel_gains, action)

        # 更新上一个状态的奖励值 和 信道增益。
        self._laststate[-1] = reward
        self._laststate[0:-1] = self.channel_gains * real_action
        # self._laststate[0:-1] = self.channel_gains * real_action
        # 增加步数。
        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        # 检查是否达到回合最大步数，如果是，则终止回合。
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        # Information about number of steps taken
        # 返回新的状态、奖励、回合是否终止以及包含步数和动作信息的字典。
        info = {'num_steps': self._num_steps, 'expert_action': expert_action, 'sub_expert_action': sub_expert_action}
        return self._laststate, reward, self._terminated, info

    def reset(self):
        # Reset the environment to its initial state
        # 重置环境，返回初始状态和包含步数信息的字典
        # 将步数重置为 0
        self._num_steps = 0
        # 将回合终止标志重置为 False。
        self._terminated = False
        # 返回初始状态和包含步数信息的字典。
        state = self.state
        return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        # Set seed for random number generation
        # 设置随机数生成器的种子，以便重现实验结果c
        np.random.seed(seed)


# 设置随机数生成器的种子，以便重现实验结果。
def make_aigc_env(training_num=0, test_num=0):
    """Wrapper function for AIGC env.
    :return: a tuple of (single env, training envs, test envs).
    """
    # 创建一个 AIGCEnv 实例并设置种子。
    # 如果 training_num 大于 0，则创建多个 AIGCEnv 实例并包装成 DummyVectorEnv 训练环境。
    # 如果 test_num 大于 0，则创建多个 AIGCEnv 实例并包装成 DummyVectorEnv 测试环境。
    # 返回单个环境实例、训练环境实例和测试环境实例。

    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    # training_num：要创建的训练环境数量。
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    # test_num：要创建的测试环境数量
    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
    #  返回一个包含单个环境实例、训练环境实例和测试环境实例的元组。
    return env, train_envs, test_envs
