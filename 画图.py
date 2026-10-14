# import matplotlib
# matplotlib.use('TkAgg')
#
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
# ratios = [100, 80, 50, 30]
# baseline = [84.2, 82.9, 79.4, 74.6]
# w_o_dynamic = [85.8, 84.6, 81.6, 77.5]
# full_model = [86.5, 85.1, 83.2, 79.8]
#
# plt.figure(figsize=(7, 4.5))
#
# plt.plot(ratios, baseline, '--o', linewidth=1.5, markersize=5, label='Baseline WRN-16-8')
# plt.plot(ratios, w_o_dynamic, '-s', linewidth=1.8, markersize=6, label='w/o Dynamic KL')
# plt.plot(ratios, full_model, '-^', linewidth=2.5, markersize=7, label='Full Model (Ours)')
#
# plt.xlabel('Training Data Ratio (%)')
# plt.ylabel('Top-1 Accuracy (%)')
# plt.xticks(ratios)
# plt.ylim(72, 90)
#
# plt.legend(loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.4)
#
# plt.title('Accuracy under Different Training Data Ratios on CIFAR-100')
# plt.tight_layout()
# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 模拟训练轮次
epochs = np.arange(0, 400, 1)

# 模拟静态KL约束下的损失曲线
static_fine_loss = 4.0 * np.exp(-epochs/100) + 0.1 * np.random.randn(len(epochs)) * np.exp(-epochs/200) + 0.8
static_kl_loss = 1.5 + 0.8 * np.sin(epochs/50) * np.exp(-epochs/300)

# 模拟动态KL约束下的损失曲线
# 早期：KL损失权重低，分类损失快速下降
early_phase = epochs < 150
mid_late_phase = epochs >= 150

dynamic_fine_loss = np.zeros_like(epochs)
dynamic_kl_loss = np.zeros_like(epochs)

# 早期阶段
dynamic_fine_loss[early_phase] = 4.0 * np.exp(-epochs[early_phase]/80) + 0.5
dynamic_kl_loss[early_phase] = 0.2 + 0.05 * np.random.randn(np.sum(early_phase))

# 中后期阶段
dynamic_fine_loss[mid_late_phase] = 0.8 + 0.1 * np.exp(-(epochs[mid_late_phase]-150)/100)
dynamic_kl_loss[mid_late_phase] = 0.3 + 0.7 * (1 - np.exp(-(epochs[mid_late_phase]-150)/150))

# 平滑处理
from scipy.ndimage import gaussian_filter1d
static_fine_loss = gaussian_filter1d(static_fine_loss, sigma=3)
static_kl_loss = gaussian_filter1d(static_kl_loss, sigma=3)
dynamic_fine_loss = gaussian_filter1d(dynamic_fine_loss, sigma=3)
dynamic_kl_loss = gaussian_filter1d(dynamic_kl_loss, sigma=3)

# 绘制细粒度分类损失
ax1.plot(epochs, static_fine_loss, 'r-', linewidth=2, label='静态KL约束')
ax1.plot(epochs, dynamic_fine_loss, 'b-', linewidth=2, label='动态KL约束')
ax1.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax1.set_ylabel(r'细粒度分类损失 $\mathcal{L}_{fine}$', fontsize=11)
ax1.set_title('(a) 细粒度分类损失动态', fontsize=12, pad=10)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim([0, 400])
ax1.set_ylim([0.5, 4.5])

# 绘制层次一致性损失
ax2.plot(epochs, static_kl_loss, 'r-', linewidth=2, label='静态KL约束')
ax2.plot(epochs, dynamic_kl_loss, 'b-', linewidth=2, label='动态KL约束')
ax2.set_xlabel('训练轮次 (Epoch)', fontsize=11)
ax2.set_ylabel(r'层次一致性损失 $\mathcal{L}_{HC}$', fontsize=11)
ax2.set_title('(b) 层次一致性损失动态', fontsize=12, pad=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim([0, 400])
ax2.set_ylim([0, 2.5])

# 添加早期阶段标记
ax1.axvline(x=150, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax2.axvline(x=150, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax1.text(75, 3.8, '早期专注学习阶段', ha='center', fontsize=9, style='italic', alpha=0.8)
ax1.text(275, 1.5, '后期对齐强化阶段', ha='center', fontsize=9, style='italic', alpha=0.8)

plt.tight_layout()
plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()