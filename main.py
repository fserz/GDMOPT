# Import necessary libraries
import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Define a function to get command line arguments
def get_args():
    # Create argument parser
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加探索噪声参数，默认值为0.1
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    # 添加算法类型参数，默认值为'diffusion_opt'
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    # 添加随机种子参数，默认值为1
    parser.add_argument('--seed', type=int, default=1)
    # 添加缓冲区大小参数，默认值为1e6（100万）
    # parser.add_argument('--buffer-size', type=int, default=1e6)#1e6
    parser.add_argument('--buffer-size', type=int, default=3000)#1e6
    # 添加训练周期数参数，默认值为1e6（100万）
    # parser.add_argument('-e', '--epoch', type=int, default=1e6)# 1000
    parser.add_argument('-e', '--epoch', type=int, default=3000)# 1000
    # 添加每个周期的步数参数，默认值为1
    parser.add_argument('--step-per-epoch', type=int, default=1)# 100
    # 添加每次收集的步数参数，默认值为1
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    # 添加批处理大小参数，默认值为512
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    # 添加权重衰减参数，默认值为1e-4
    parser.add_argument('--wd', type=float, default=1e-4)
    # 添加折扣因子参数，默认值为1
    parser.add_argument('--gamma', type=float, default=1)
    # 添加n步TD学习参数，默认值为3
    parser.add_argument('--n-step', type=int, default=3)
    # 添加训练环境数量参数，默认值为1
    parser.add_argument('--training-num', type=int, default=1)
    # 添加测试环境数量参数，默认值为1
    parser.add_argument('--test-num', type=int, default=1)
    # 添加日志目录参数，默认值为'log'
    parser.add_argument('--logdir', type=str, default='log')
    # 添加日志前缀参数，默认值为'default'
    parser.add_argument('--log-prefix', type=str, default='default')
    # 添加渲染参数，默认值为0.1
    parser.add_argument('--render', type=float, default=0.1)
    # 添加奖励归一化参数，默认值为0
    parser.add_argument('--rew-norm', type=int, default=0)
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # 添加设备参数，默认值为'cuda:0'
    parser.add_argument(
        '--device', type=str, default='cuda:0')
    # 添加恢复路径参数，默认值为None
    parser.add_argument('--resume-path', type=str, default=None)
    # 添加观看模式参数，默认为False
    parser.add_argument('--watch', action='store_true', default=False)
    # 添加学习率衰减参数，默认为False
    parser.add_argument('--lr-decay', action='store_true', default=False)
    # 添加备注参数，默认值为空字符串
    parser.add_argument('--note', type=str, default='')

    # for diffusion
    # 添加演员学习率参数，默认值为1e-4
    # parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--actor-lr', type=float, default=1e-2)
    # 添加评论家学习率参数，默认值为1e-4
    # parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-2)
    # 添加软更新参数tau，默认值为0.005
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    # 添加时间步数参数，默认值为6
    parser.add_argument('-t', '--n-timesteps', type=int, default=6)  # for diffusion chain 3 & 8 & 12
    # 添加beta调度参数，默认值为'vp'，可选值为'linear'、'cosine'、'vp'
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])

    # With Expert: bc-coef True
    # Without Expert: bc-coef False
    # parser.add_argument('--bc-coef', default=False) # Apr-04-132705
    # 添加行为克隆系数参数，默认值为False
    parser.add_argument('--bc-coef', default=False)

    # for prioritized experience replay
    # 添加优先经验回放参数，默认为False
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    # 添加优先级回放的alpha参数，默认值为0.4
    parser.add_argument('--prior-alpha', type=float, default=0.4)#
    # 添加优先级回放的 beta 参数，默认值为 0.4
    parser.add_argument('--prior-beta', type=float, default=0.4)#

    # Parse arguments and return them
    # 解析命令行参数并返回
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environments
    # 创建环境，然后设置状态和动作的形状以及最大动作值。
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.n
    args.max_action = 1.

    # 调整探索噪声的幅度。
    args.exploration_noise = args.exploration_noise * args.max_action
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # create actor
    # 创建一个多层感知器（MLP）网络，用于 actor。
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )
    # Actor is a Diffusion model
    # Actor 是一个 Diffusion 模型
    # 使用 Diffusion 模型构建 actor，并定义其优化器。
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        bc_coef = args.bc_coef
    ).to(args.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # Create critic
    # 创建双重评论器（Double Critic）和对应的优化器。
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    ## Setup logging
    # 设置日志记录
    # 设置日志记录，创建日志路径，并初始化 TensorBoard 记录器。
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # def dist(*logits):
    #    return Independent(Normal(*logits), 1)

    # Define policy
    # 定义策略
    # 定义策略 DiffusionOPT，并将必要参数传递给它
    policy = DiffusionOPT(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        # dist,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        bc_coef=args.bc_coef,
        action_space=env.action_space,
        exploration_noise = args.exploration_noise,
    )

    # Load a previous policy if a path is provided
    # 如果提供了路径，则加载之前的策略
    # 如果提供了恢复路径，则从该路径加载之前保存的策略参数。
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # Setup buffer
    # 设置经验回放缓冲区
    # 根据是否使用优先经验回放，创建对应的经验回放缓冲区。
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    # 设置收集器
    # 创建训练和测试数据的收集器。
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # 定义一个函数，用于保存表现最好的策略参数。
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Trainer
    # 训练器
    # 检查是否需要进入“观看”模式（即只评估模型，不进行训练）。如果 args.watch 为 False，则进行训练。
    # 调用 offpolicy_trainer 函数来训练策略。传递的参数包括：
        # policy：训练的策略。
        # train_collector：用于收集训练数据的收集器。
        # test_collector：用于收集测试数据的收集器。
        # args.epoch：训练的总轮数。
        # args.step_per_epoch：每轮训练的步数。
        # args.step_per_collect：每次收集的步数。
        # args.test_num：测试环境的数量。
        # args.batch_size：训练的批量大小。
        # save_best_fn：用于保存最佳策略的函数。
        # logger：日志记录器。
        # test_in_train：在训练过程中是否进行测试
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
        )
        # 打印训练结果。
        pprint.pprint(result)

    # Watch the performance
    # python main.py --watch --resume-path log/default/diffusion/Jul10-142653/policy.pth
    if __name__ == '__main__':
        # 将策略设置为评估模式。
        policy.eval()
        # 创建一个新的收集器，用于评估策略的表现。
        collector = Collector(policy, env)
        # 收集一个完整的评估回合，参数 n_episode=1 表示只评估一个回合。可以取消注释 render=args.render 来渲染环境。
        result = collector.collect(n_episode=1) #, render=args.render
        # 打印评估结果。
        print(result)
        # 提取评估回合的奖励和长度。
        rews, lens = result["rews"], result["lens"]
        # 打印最终的平均奖励和平均长度。
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
