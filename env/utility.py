import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 定义一个名为 rayleigh_channel_gain 的函数，它有两个参数 ex 和 sta。这两个参数通常表示正态分布的期望值（均值）和标准差
def rayleigh_channel_gain(ex, sta):
    # 设置采样数量 num_samples 为 1，这意味着我们只生成一个样本。
    num_samples = 1
    # 使用 numpy 的 np.random.normal 函数生成一个服从正态分布（高斯分布）的随机样本，均值为 ex，标准差为 sta，样本数量为 num_samples。
    gain = np.random.normal(ex, sta, num_samples)
    # Square the absolute value to get Rayleigh-distributed gains
    # 通过取绝对值并平方来获得瑞利分布的增益。
    # 计算生成样本的绝对值，并将其平方。对于正态分布的随机变量，这样的操作会生成一个瑞利分布的随机变量。
    gain = np.abs(gain) ** 2
    return gain

# ****************原论文代码***************

# Function to implement water filling algorithm for power allocation
# 实现功率分配注水算法的函数
# def water(s, total_power):
#     a = total_power
#     # Define the channel gain and noise level
#     # 定义信道增益 g_n，并假设噪声水平 N_0 为 1
#     g_n = s
#     # 假设所有传输的噪声级固定为 1，这可以根据您的要求进行更改
#     N_0 = 1  # Assuming a fixed noise-level of 1 for all transmissions, this can be changed based on your requirement
#
#     # Initialize the upper and lower bounds for the bisection search
#     # 初始化二分搜索的上界和下界
#     L = 0
#     U = a + N_0 * np.sum(1 / (g_n + 1e-6))  # Initial guess for upper bound
#
#     # Define the precision for the bisection search
#     # 定义二分搜索的精度为 1e-6
#     precision = 1e-6
#
#     # Perform the bisection search for the power level
#     # 对功率级别进行二分搜索，当上下界的差值大于精度时继续循环。
#     while U - L > precision:
#         # 将当前级别设置为边界中间 alpha_bar
#         alpha_bar = (L + U) / 2  # Set the current level to be in the middle of bounds
#         # 计算功率分配，并确保结果为非负值。
#         p_n = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)  # Calculate the power allocation
#         # 计算总功率，即所有信道上的功率分配之和。
#         P = np.sum(p_n)  # Calculate the total power
#
#         # 检查功率预算是否利用不足或过度
#         # Check whether the power budget is under or over-utilized
#         # 如果功率预算过度利用
#         if P > a:  # If the power budget is over-utilized
#             # 将上限移至当前功率级别
#             U = alpha_bar  # Move the upper bound to the current power level
#             # 如果功率水平低于功率预算
#         else:  # If the power level is below the power budget
#             # 上移下限
#             L = alpha_bar  # Move the lower bound up
#
#     # 计算最终的功率分配
#     # Calculate the final power allocation
#     p_n_final = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)
#
#     # 计算每个通道的数据速率
#     # Calculate the data rate for each channel
#     # 计算信噪比
#     SNR = g_n * p_n_final / N_0  # Calculate the SNR
#     # 计算数据速率
#     data_rate = np.log2(1 + SNR)  # Calculate the data rate
#     sumdata_rate = np.sum(data_rate)
#     # print('p_n_final', p_n_final)
#     # print('data_rate', sumdata_rate)
#     expert = p_n_final / total_power
#     subexpert = p_n_final / total_power + np.random.normal(0, 0.1, len(p_n_final))
#     return expert, sumdata_rate, subexpert
#
# # 定义一个名为 CompUtility 的函数，用于计算给定状态和动作的效用（奖励）。
# # Function to compute utility (reward) for the given state and action
# def CompUtility(State, Aution):
#     # 将 Aution 转换为 NumPy 数组，然后再转换为 PyTorch 的 FloatTensor 类型，方便后续的张量操作。
#     actions = torch.from_numpy(np.array(Aution)).float()
#     # 对动作取绝对值，确保所有动作值为非负数。
#     actions = torch.abs(actions)
#     # actions = torch.sigmoid(actions)
#     # 将 PyTorch 张量重新转换为 NumPy 数组。
#     Aution = actions.numpy()
#     # 设置总功率 total_power 为 3。这是一个常数，表示可以分配的总功率。
#     total_power = 3
#     # 将动作进行归一化处理，使得所有动作值的和为 1。这一步将动作值转换为权重。
#     normalized_weights = Aution / np.sum(Aution)
#     # 将归一化后的权重乘以总功率，得到实际的功率分配 a。
#     a = normalized_weights * total_power
#
#     # 将 State 赋值给 g_n，表示信道增益。
#     g_n = State
#     # 计算信噪比（SNR），公式为 SNR = 信道增益 * 分配的功率。
#     SNR = g_n * a
#
#     # 计算数据速率，公式为 data_rate = log2(1 + SNR)。这个公式来自香农公式，用于计算在给定信噪比下的最大数据传输速率。
#     data_rate = np.log2(1 + SNR)
#
#     # 调用 water 函数，用于计算专家的动作（即最优的功率分配方案），返回专家的动作，专家动作的总数据速率，以及子最优专家动作。
#     expert_action, sumdata_rate, subopt_expert_action = water(g_n, total_power)
#
#     # 计算奖励，公式为 reward = 当前动作的数据速率总和 - 专家动作的数据速率总和。这个奖励表示当前动作相对于专家动作的性能差异。
#     reward = np.sum(data_rate) - sumdata_rate
#     # reward = np.sum(data_rate) - sumdata_rate
#
#     # 返回计算得到的奖励、专家的动作、子最优专家的动作以及归一化后的动作值。
#     return reward, expert_action, subopt_expert_action, Aution

# ****************原论文代码***************



# ********
# Function to compute utility (reward) for the given state and action
# def CompUtility(State, Aution):
#     actions = torch.from_numpy(np.array(Aution)).float()
#     actions = torch.abs(actions)
#     Aution = actions.numpy()
#     # 总负载
#     total_load = 1305
#     normalized_weights = Aution / np.sum(Aution)
#     # 实际分到的负载
#     load_allocation = normalized_weights * total_load
#
#     g_n = State
#     SNR = g_n * load_allocation
#     data_rate = np.log2(1 + SNR)
#
#     expert_action, sumdata_rate, subopt_expert_action = allocate_load(g_n, total_load)
#
#     reward = np.sum(data_rate) - sumdata_rate
#
#     return reward, expert_action, subopt_expert_action, Aution
#
# # # Function to allocate load based on given transmission rates
# def allocate_load(transmission_rates, total_load):
#     n_channels = len(transmission_rates)
#     # 初始化一个大小为通道数的零数组，用于存储每个通道的负载分配
#     load_allocation = np.zeros(n_channels)
#     # 获取按传输速率从高到低排序的索引
#     sorted_indices = np.argsort(transmission_rates)[::-1]
#     # 按从高到低的顺序排列传输速率
#     sorted_rates = transmission_rates[sorted_indices]
#
#     # 初始化二分查找的上下界
#     L = 0
#     U = total_load / min(sorted_rates)
#     # 设置二分查找的精度
#     precision = 1e-6
#
#     # 当上下界的差值大于精度时继续二分查找
#     while U - L > precision:
#         alpha = (L + U) / 2
#         tentative_allocation = np.minimum(total_load, alpha * sorted_rates)
#         total_allocated_load = np.sum(tentative_allocation)
#
#         if total_allocated_load < total_load:
#             L = alpha
#         else:
#             U = alpha
#
#     load_allocation[sorted_indices] = np.minimum(total_load, alpha * sorted_rates)
#
#     # Calculate data rate
#     SNR = transmission_rates * load_allocation
#     data_rate = np.log2(1 + SNR)
#     sumdata_rate = np.sum(data_rate)
#
#     # Generate expert action and suboptimal expert action
#     expert_action = load_allocation / total_load
#     subopt_expert_action = load_allocation / total_load + np.random.normal(0, 0.1, len(load_allocation))
#
#     # return expert_action, sumdata_rate, subopt_expert_action, load_allocation
#     # 返回专家动作、总数据速率和次优专家动作
#     return expert_action, sumdata_rate, subopt_expert_action

# ==============================原来的对比算法==========================
# 和之前结果一样
# def CompUtility(State, Aution):
#     actions = torch.from_numpy(np.array(Aution)).float()
#     actions = torch.abs(actions)
#     Aution = actions.numpy()
#     # 总负载
#     total_load = 1305
#     normalized_weights = Aution / np.sum(Aution)
#     # 实际分到的负载
#     load_allocation = normalized_weights * total_load
#
#     g_n = State
#     SNR = g_n * load_allocation
#     data_rate = np.log2(1 + SNR)
#
#     expert_action, sumdata_rate, subopt_expert_action = allocate_load(g_n, total_load)
#
#     reward = np.sum(data_rate) - sumdata_rate
#
#     return reward, expert_action, subopt_expert_action, Aution
#
# # Function to allocate load based on given transmission rates
# def allocate_load(transmission_rates, total_load):
#     n_channels = len(transmission_rates)
#     # 初始化一个大小为通道数的零数组，用于存储每个通道的负载分配
#     load_allocation = np.zeros(n_channels)
#     # 获取按传输速率从高到低排序的索引
#     sorted_indices = np.argsort(transmission_rates)[::-1]
#     # 按从高到低的顺序排列传输速率
#     sorted_rates = transmission_rates[sorted_indices]
#
#     # 初始化二分查找的上下界
#     L = 0
#     U = total_load / min(sorted_rates)
#     # 设置二分查找的精度
#     precision = 1e-6
#
#     # 当上下界的差值大于精度时继续二分查找
#     while U - L > precision:
#         alpha = (L + U) / 2
#         tentative_allocation = np.minimum(total_load, alpha * sorted_rates)
#         total_allocated_load = np.sum(tentative_allocation)
#
#         if total_allocated_load < total_load:
#             L = alpha
#         else:
#             U = alpha
#
#     load_allocation[sorted_indices] = np.minimum(total_load, alpha * sorted_rates)
#
#     # Calculate data rate
#     SNR = transmission_rates * load_allocation
#     data_rate = np.log2(1 + SNR)
#     sumdata_rate = np.sum(data_rate)
#
#     # Generate expert action and suboptimal expert action
#     expert_action = load_allocation / total_load
#     subopt_expert_action = load_allocation / total_load + np.random.normal(0, 0.1, len(load_allocation))
#
#     # 返回专家动作、总数据速率和次优专家动作
#     return expert_action, sumdata_rate, subopt_expert_action
#
# # Function to allocate load based on random transmission rates
# def allocate_load_random(total_load, num_channels=5, rate_range=(10, 30)):
#     # 随机生成传输速率
#     transmission_rates = np.random.uniform(rate_range[0], rate_range[1], num_channels)
#     return allocate_load(transmission_rates, total_load)

# ==================================================================

#
# # ======================================新的我们的算法================================
# def CompUtility(State, Aution):
#     # 转换动作为张量并取绝对值
#     actions = torch.from_numpy(np.array(Aution)).float()
#     actions = torch.abs(actions)
#     Aution = actions.numpy()
#     # 计算归一化权重并分配负载
#     total_load = 1305
#     normalized_weights = Aution / np.sum(Aution)
#     load_allocation = normalized_weights * total_load
#
#     # 计算信噪比 (SNR) 和数据速率
#     g_n = State
#     SNR = g_n * load_allocation
#     data_rate = np.log2(1 + SNR)
#
#     # 使用传输速率而不是信道增益
#     # 调用 allocate_load 函数计算专家策略
#     transmission_rates = np.array([60, 40, 10, 20, 30])
#     expert_action, expert_data_rate, subopt_expert_action = allocate_load(transmission_rates, total_load)
#
#     # 计算奖励,奖励定义为动态分配策略的数据速率之和减去专家策略的数据速率
#     reward = np.sum(data_rate) - expert_data_rate
#
#     # ****************************************
#     # 计算总时延
#     total_data_rate = np.sum(data_rate)
#     total_delay = total_load / total_data_rate
#
#     # 输出负载分配和总时延
#     print("Load Allocation:", load_allocation)
#     print("Total Delay:", total_delay)
#     # ****************************************
#
#     # 返回奖励、专家动作、次优专家动作以及归一化后的动作。
#     return reward, expert_action, subopt_expert_action, Aution
#
# # transmission_rates：每个信道的传输速率。 total_load：总负载。
# def allocate_load(transmission_rates, total_load):
#     # 初始化和排序,初始化负载分配并对传输速率进行降序排序
#     n_channels = len(transmission_rates)
#     load_allocation = np.zeros(n_channels)
#     sorted_indices = np.argsort(transmission_rates)[::-1]
#     sorted_rates = transmission_rates[sorted_indices]
#
#     # 二分搜索确定 alpha ,通过二分搜索法确定适合的 alpha 值，使得总分配负载等于 total_load
#     L = 0
#     U = total_load / min(sorted_rates)
#     precision = 1e-6
#
#     while U - L > precision:
#         alpha = (L + U) / 2
#         tentative_allocation = np.minimum(total_load, alpha * sorted_rates)
#         total_allocated_load = np.sum(tentative_allocation)
#
#         if total_allocated_load < total_load:
#             L = alpha
#         else:
#             U = alpha
#
#     # 根据找到的alpha值，分配负载。
#     load_allocation[sorted_indices] = np.minimum(total_load, alpha * sorted_rates)
#
#     # 计算信噪比和数据速率，并计算总数据速率。
#     SNR = transmission_rates * load_allocation
#     data_rate = np.log2(1 + SNR)
#     sum_data_rate = np.sum(data_rate)
#
#     # 计算专家动作和次优专家动作：
#     # 专家动作是负载分配归一化结果，次优专家动作是在专家动作上加上一个高斯噪声。
#     expert_action = load_allocation / total_load
#     subopt_expert_action = load_allocation / total_load + np.random.normal(0, 0.1, len(load_allocation))
#
#     # 返回专家动作、总数据速率和次优专家动作。
#     return expert_action, sum_data_rate, subopt_expert_action


# =========================随机生成5个信道速率=================================================
# def CompUtility(State, Aution):
#     # 转换动作为张量并取绝对值
#     actions = torch.from_numpy(np.array(Aution)).float()
#     actions = torch.abs(actions)
#     Aution = actions.numpy()
#     # 计算归一化权重并分配负载
#     total_load = 1305
#     normalized_weights = Aution / np.sum(Aution)
#     load_allocation = normalized_weights * total_load
#
#     # 计算信噪比 (SNR) 和数据速率
#     g_n = State
#     SNR = g_n * load_allocation
#     data_rate = np.log2(1 + SNR)
#
#     # 使用随机生成的传输速率而不是固定的传输速率
#     # 调用 allocate_load 函数计算专家策略
#     expert_action, expert_data_rate, subopt_expert_action = allocate_load(total_load)
#
#     # 计算奖励, 奖励定义为动态分配策略的数据速率之和减去专家策略的数据速率
#     reward = np.sum(data_rate) - expert_data_rate
#
#     # # ****************************************
#     # # 计算总时延
#     # total_data_rate = np.sum(data_rate)
#     # total_delay = total_load / total_data_rate
#     #
#     # # 输出负载分配和总时延
#     # print("Load Allocation:", load_allocation)
#     # print("Total Delay:", total_delay)
#     # # ****************************************
#
#     # 返回奖励、专家动作、次优专家动作以及归一化后的动作。
#     return reward, expert_action, subopt_expert_action, Aution
#
# # total_load：总负载。
# def allocate_load(total_load):
#     # 随机生成传输速率
#     transmission_rates = np.random.uniform(60, 100, 5)
#
#     # 初始化和排序,初始化负载分配并对传输速率进行降序排序
#     n_channels = len(transmission_rates)
#     load_allocation = np.zeros(n_channels)
#     sorted_indices = np.argsort(transmission_rates)[::-1]
#     sorted_rates = transmission_rates[sorted_indices]
#
#     # 二分搜索确定 alpha ,通过二分搜索法确定适合的 alpha 值，使得总分配负载等于 total_load
#     L = 0
#     U = total_load / min(sorted_rates)
#     precision = 1e-6
#
#     while U - L > precision:
#         alpha = (L + U) / 2
#         tentative_allocation = np.minimum(total_load, alpha * sorted_rates)
#         total_allocated_load = np.sum(tentative_allocation)
#
#         if total_allocated_load < total_load:
#             L = alpha
#         else:
#             U = alpha
#
#     # 根据找到的alpha值，分配负载。
#     load_allocation[sorted_indices] = np.minimum(total_load, alpha * sorted_rates)
#
#     # 计算信噪比和数据速率，并计算总数据速率。
#     SNR = transmission_rates * load_allocation
#     data_rate = np.log2(1 + SNR)
#     sum_data_rate = np.sum(data_rate)
#
#     # 计算专家动作和次优专家动作：
#     # 专家动作是负载分配归一化结果，次优专家动作是在专家动作上加上一个高斯噪声。
#     expert_action = load_allocation / total_load
#     subopt_expert_action = load_allocation / total_load + np.random.normal(0, 0.1, len(load_allocation))
#
#     # 返回专家动作、总数据速率和次优专家动作。
#     return expert_action, sum_data_rate, subopt_expert_action
# ==========================================================================


# ============================================随机生成一条信道速率传输===========================
# def CompUtility(State, Aution):
#     # 转换动作为张量并取绝对值
#     actions = torch.from_numpy(np.array(Aution)).float()
#     actions = torch.abs(actions)
#     Aution = actions.numpy()
#     # 计算归一化权重并分配负载
#     total_load = 1305
#     normalized_weights = Aution / np.sum(Aution)
#     load_allocation = normalized_weights * total_load
#
#     # 计算信噪比 (SNR) 和数据速率
#     g_n = State
#     SNR = g_n * load_allocation
#     data_rate = np.log2(1 + SNR)
#
#     # 使用随机生成的单个信道传输速率而不是固定的传输速率
#     # 调用 allocate_load 函数计算专家策略
#     expert_action, expert_data_rate, subopt_expert_action = allocate_load(total_load)
#
#     # 计算奖励, 奖励定义为动态分配策略的数据速率之和减去专家策略的数据速率
#     reward = np.sum(data_rate) - expert_data_rate
#
#     # ****************************************
#     # # 计算总时延
#     # total_data_rate = np.sum(data_rate)
#     # total_delay = total_load / total_data_rate
#     #
#     # # 输出负载分配和总时延
#     # print("Load Allocation:", load_allocation)
#     # print("Total Delay:", total_delay)
#     # ****************************************
#
#     # 返回奖励、专家动作、次优专家动作以及归一化后的动作。
#     return reward, expert_action, subopt_expert_action, Aution

# total_load：总负载。
# def allocate_load(total_load):
#     # 随机生成一个信道的传输速率
#     transmission_rate = np.random.uniform(10, 20, 1)[0]
#
#     # 初始化负载分配
#     load_allocation = np.zeros(1)
#
#     # 直接分配所有负载到这个单一信道
#     load_allocation[0] = total_load
#
#     # 计算信噪比和数据速率
#     SNR = transmission_rate * load_allocation
#     data_rate = np.log2(1 + SNR)
#     sum_data_rate = np.sum(data_rate)
#
#     # 计算专家动作和次优专家动作：
#     # 专家动作是负载分配归一化结果，次优专家动作是在专家动作上加上一个高斯噪声。
#     expert_action = load_allocation / total_load
#     subopt_expert_action = load_allocation / total_load + np.random.normal(0, 0.1, len(load_allocation))
#
#     # 返回专家动作、总数据速率和次优专家动作。
#     return expert_action, sum_data_rate, subopt_expert_action
#
# # ==============================轮询=======================================
# 定义全局变量，用于记录上次选择的信道索引
last_channel_index = 0

def CompUtility(State, Aution):
    global last_channel_index

    # 转换动作为张量并取绝对值
    actions = torch.from_numpy(np.array(Aution)).float()
    actions = torch.abs(actions)
    Aution = actions.numpy()
    # 计算归一化权重并分配负载
    total_load = 1305
    normalized_weights = Aution / np.sum(Aution)
    load_allocation = normalized_weights * total_load

    # 计算信噪比 (SNR) 和数据速率
    g_n = State
    SNR = g_n * load_allocation
    data_rate = np.log2(1 + SNR)

    # 使用传输速率而不是信道增益
    # 获取下一个信道的传输速率
    transmission_rates = np.array([60, 40, 10, 20, 30])
    channel_index = last_channel_index % len(transmission_rates)
    expert_action, expert_data_rate, subopt_expert_action = allocate_load(transmission_rates[channel_index], total_load)

    # 计算奖励, 奖励定义为动态分配策略的数据速率之和减去专家策略的数据速率
    reward = np.sum(data_rate) - expert_data_rate

    # 更新上次选择的信道索引
    last_channel_index += 1

    # # ****************************************
    # # 计算总时延
    # total_data_rate = np.sum(data_rate)
    # total_delay = total_load / total_data_rate
    #
    # # 输出负载分配和总时延
    # print("Load Allocation:", load_allocation)
    # print("Total Delay:", total_delay)
    # # ****************************************

    # 返回奖励、专家动作、次优专家动作以及归一化后的动作。
    return reward, expert_action, subopt_expert_action, Aution

# 返回一个或多个信道传输速率,总载荷
def allocate_load(transmission_rate, total_load):
    # 初始化分配载荷
    load_allocation = np.zeros(1)
    load_allocation[0] = total_load

    # 计算SNR和数据率
    SNR = transmission_rate * load_allocation
    data_rate = np.log2(1 + SNR)
    sum_data_rate = np.sum(data_rate)

    # 计算专家动作和次优专家动作
    expert_action = load_allocation / total_load
    subopt_expert_action = expert_action + np.random.normal(0, 0.1, len(load_allocation))

    # 返回专家动作、总数据速率和次优专家动作
    return expert_action, sum_data_rate, subopt_expert_action


# # Example usage
# transmission_rates = np.array([60, 40, 10, 20, 30])  # Transmission rates for each channel
# total_load = 1305  # Total load to be distributed
#
# reward, expert_action, subopt_expert_action, load_allocation = CompUtility(transmission_rates, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
# print(f"Reward: {reward}")
# print(f"Expert Action: {expert_action}")
# print(f"Suboptimal Expert Action: {subopt_expert_action}")
# print(f"Load Allocation: {load_allocation}")