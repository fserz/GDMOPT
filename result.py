import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# 文件路径列表
csv_files = ['CSV/log_default_diffusion_Jul02-234336.csv', 'CSV/log_default_diffusion_Jul03-092026.csv', 'CSV/log_default_diffusion_Jul04-002415.csv']

# 预先定义标签
labels = ['Random 5', 'Random one', 'Polling']

plt.figure()

# 遍历每个文件和标签，绘制曲线
for file, label in zip(csv_files, labels):
    # 从CSV文件加载数据
    df = pd.read_csv(file)

    # 提取数据列（注意列名的大小写）
    steps = df['Step']
    values = df['Value']

    # 绘制曲线
    plt.plot(steps, values, label=label)

# 添加图例
plt.xlabel('Step (millions)')
plt.ylabel('Value')
plt.title('Experiment Results')
plt.legend()
plt.grid(True)

# 定义横坐标刻度格式化函数
formatter = FuncFormatter(lambda x, _: f'{x / 1e6:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)

plt.savefig('experiment_results.png')
plt.show()
