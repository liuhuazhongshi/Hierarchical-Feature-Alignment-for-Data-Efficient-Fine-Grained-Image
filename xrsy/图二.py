#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成符合 SCI 论文标准的 Grad-CAM 可视化对比图
寻找 baseline 错误但 HFENet 正确的样本
用于论文 Figure 3
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import cv2
import os
from collections import OrderedDict
import matplotlib.font_manager as fm

# 尝试设置字体，如果失败则使用默认 serif
def set_sci_fonts():
    plt.rcParams['font.family'] = 'serif'
    # 尝试指定 Times New Roman，如果系统没有，matplotlib 会自动找其他 serif 字体
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIXGeneral', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix' if 'STIXGeneral' in [f.name for f in fm.fontManager.ttflist] else 'cm'
    plt.rcParams['font.size'] = 10  # 稍微加大字号以防看不清
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = None # 关闭自动 bbox，我们手动控制
    plt.rcParams['savefig.pad_inches'] = 0.1

set_sci_fonts()
# 尝试导入 base_experiment，如果不存在请确保该文件在当前目录
try:
    from base_experiment import WideResNet
except ImportError:
    print("❌ 错误: 未找到 'base_experiment.py' 或 'WideResNet' 类。请确保文件存在。")


    # 为了演示代码运行，这里定义一个空的占位类，实际使用时请注释掉以下两行并修复导入
    class WideResNet:
        def __init__(self, **kwargs): pass

        def to(self, device): return self

        def eval(self): pass

        def zero_grad(self): pass

# ========== 配置区域 ==========
CHECKPOINT_PATH = "./ckpt/exp_baseline_latest.pth"  # 模型检查点路径
OUTPUT_DIR = "./imgs_sci_ready"  # 输出目录
NUM_SAMPLES_TO_FIND = 3  # 需要展示的样本数量
# ============================

# CIFAR-100 类别名称 (保持原样)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# ========== SCI 论文绘图配置 ==========
# 设置全局字体为 Times New Roman (LaTeX 标准)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 如果没有 Times New Roman，matplotlib 可能会回退，确保系统安装了该字体
# 对于数学符号，使用 LaTeX 渲染或内置 mathtext
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9  # 正文字号
plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线宽
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['figure.dpi'] = 300  # 屏幕显示 DPI
plt.rcParams['savefig.dpi'] = 600  # 保存 DPI (SCI 通常要求 300-600)
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05  # 保留少量白边，防止裁剪


class GradCAM:
    """Grad-CAM 实现 (保持逻辑不变，优化异常处理)"""

    def __init__(self, model, target_layer, is_baseline=False):
        self.model = model
        self.target_layer = target_layer
        self.is_baseline = is_baseline
        self.gradients = None
        self.activations = None

        # 注册 Hook
        self.hook_activation = target_layer.register_forward_hook(self.save_activation)
        self.hook_gradient = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image):
        self.model.zero_grad()
        try:
            if self.is_baseline:
                logit_f = self.model(input_image.unsqueeze(0))
                pred_class = logit_f.argmax(dim=1).item()
                one_hot = torch.zeros_like(logit_f)
                one_hot[0, pred_class] = 1
                logit_f.backward(gradient=one_hot, retain_graph=True)
            else:
                # 假设 HFENet 返回多个值，取第二个作为 fine logit
                outputs = self.model(input_image.unsqueeze(0))
                if isinstance(outputs, tuple):
                    logit_s, logit_f, _, _ = outputs
                else:
                    logit_f = outputs  # 兼容单输出情况
                pred_class = logit_f.argmax(dim=1).item()
                one_hot = torch.zeros_like(logit_f)
                one_hot[0, pred_class] = 1
                logit_f.backward(gradient=one_hot, retain_graph=True)

            gradients = self.gradients
            activations = self.activations

            if gradients is None or activations is None:
                return None, None

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # 插值到原图大小 (32x32)
            cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # 归一化
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            # 轻微高斯模糊使热力图更平滑 (可选，SCI 图中常见)
            cam = cv2.GaussianBlur(cam, (3, 3), 0)

            return cam, pred_class
        except Exception as e:
            print(f"Grad-CAM 生成失败：{e}")
            return None, None

    def remove_hooks(self):
        self.hook_activation.remove()
        self.hook_gradient.remove()


def get_target_layer(model):
    """获取目标卷积层 (根据 WideResNet 结构调整)"""
    # 注意：请根据您的实际模型结构确认 layer3[-1].conv2 是否存在
    if hasattr(model, 'module'):
        m = model.module
    else:
        m = model

    # 防御性编程：检查属性是否存在
    if hasattr(m, 'layer3') and len(m.layer3) > 0:
        if hasattr(m.layer3[-1], 'conv2'):
            return m.layer3[-1].conv2
        elif hasattr(m.layer3[-1], 'conv1'):  # 备用方案
            return m.layer3[-1].conv1
    raise AttributeError("无法在模型中找到目标卷积层 (layer3[-1].conv2)。请检查模型结构。")


def load_model(checkpoint_path, device):
    print(f"\n🤖 加载模型：{checkpoint_path}")

    # 实例化两个模型结构
    model_baseline = WideResNet(depth=16, widen=8, dropout=0.1, use_coarse_branch=False, use_cross_attention=False,
                                use_kl=False).to(device)
    model_hfenet = WideResNet(depth=16, widen=8, dropout=0.1, use_coarse_branch=True, use_cross_attention=True,
                              use_kl=True).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 清理 state_dict 键名 (去除 'module.' 前缀)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # 加载权重 (strict=False 允许部分不匹配，适应双分支结构差异)
        model_baseline.load_state_dict(new_state_dict, strict=False)
        model_hfenet.load_state_dict(new_state_dict, strict=False)

        model_baseline.eval()
        model_hfenet.eval()
        print("✅ 模型加载成功")
        return model_baseline, model_hfenet
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        import traceback
        traceback.print_exc()
        return None, None


def find_error_samples(model_baseline, model_hfenet, test_loader, device, num_samples=3):
    print("\n🔍 正在筛选样本 (Baseline 错 & HFENet 对)...")
    samples = []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            # Baseline 预测
            out_b = model_baseline(x)
            pred_b = out_b.argmax(dim=1).item()

            # HFENet 预测
            out_h = model_hfenet(x)
            if isinstance(out_h, tuple):
                pred_h = out_h[1].argmax(dim=1).item()  # 取 fine logit
            else:
                pred_h = out_h.argmax(dim=1).item()

            true_label = y.item()

            if pred_b != true_label and pred_h == true_label:
                cls_true = CIFAR100_CLASSES[true_label]
                cls_pred_b = CIFAR100_CLASSES[pred_b]
                cls_pred_h = CIFAR100_CLASSES[pred_h]

                print(f"  [Found] GT:{cls_true} | Base:{cls_pred_b} (✗) | Ours:{cls_pred_h} (✓)")
                samples.append((x[0].cpu(), true_label, pred_b, pred_h))

                if len(samples) >= num_samples:
                    break

    if len(samples) < num_samples:
        print(f"⚠️ 仅找到 {len(samples)} 个符合条件的样本，将继续使用这些样本。")

    return samples


def apply_colormap_sci(cam, image_tensor):
    """
    生成符合 SCI 标准的热力图叠加图
    使用更平滑的混合和专业的颜色映射
    """
    # 1. 调整 CAM 大小
    cam_resized = cv2.resize(cam, (32, 32), interpolation=cv2.INTER_LANCZOS4)

    # 2. 颜色映射
    # 选项 A: JET (经典但有色带)，选项 B: TURBO (更平滑)，选项 C: VIRIDIS (科学友好)
    # 这里为了视觉效果接近传统热力图但更优，使用 cv2.COLORMAP_JET 后做一点处理，
    # 或者推荐使用 matplotlib 的 turbo  colormap 转换。
    # 为了简单且效果好，继续使用 JET 但增加 alpha 混合的平滑度。
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # OpenCV 默认 BGR

    # 3. 还原图像
    img_np = image_tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)

    # 4. 叠加 (Alpha Blending)
    # SCI 图中通常希望看清原图细节，alpha 设为 0.4-0.5 比较合适
    alpha = 0.45
    superimposed = cv2.addWeighted(heatmap_color, alpha, img_uint8, 1 - alpha, 0)

    return img_uint8, superimposed


def plot_gradcam_comparison(samples, model_baseline, model_hfenet, device, save_dir):
    print("\n🎨 正在绘制 SCI 标准图表 (优化排版版)...")

    # 获取层
    try:
        target_layer_b = get_target_layer(model_baseline)
        target_layer_h = get_target_layer(model_hfenet)
    except AttributeError as e:
        print(f"❌ 错误：{e}")
        return

    gradcam_b = GradCAM(model_baseline, target_layer_b, is_baseline=True)
    gradcam_h = GradCAM(model_hfenet, target_layer_h, is_baseline=False)

    n = len(samples)
    if n == 0:
        print("❌ 没有样本可绘制")
        return

    # --- 1. 精确计算画布尺寸 ---
    # 单图尺寸 (英寸): CIFAR-100 图片较小，但为了看清细节，设为 1.6 英寸高
    img_height = 1.6
    img_width = 1.6

    # 间距设置
    col_gap = 0.15  # 列间距
    row_gap = 0.1  # 行间距

    # 文本空间预留
    top_margin = 0.6  # 顶部标题空间
    bottom_margin = 0.8  # 底部标签空间 (CIFAR-100 类别名较长，需预留更多)

    # 总尺寸计算
    fig_width = 3 * img_width + 2 * col_gap
    fig_height = n * img_height + (n - 1) * row_gap + top_margin + bottom_margin

    fig, axes = plt.subplots(n, 3, figsize=(fig_width, fig_height))
    if n == 1:
        axes = axes.reshape(1, -1)

    # --- 2. 手动精确控制布局 (替代 tight_layout) ---
    # 计算归一化坐标 [0, 1]
    left = 0.08
    right = 0.95
    top = 1.0 - (top_margin / fig_height)
    bottom = bottom_margin / fig_height

    plt.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=col_gap / img_width,  # 相对宽度的间距
        hspace=row_gap / img_height  # 相对高度的间距
    )

    # --- 3. 绘制循环 ---
    for idx, (img_tensor, true_lbl, pred_b, pred_h) in enumerate(samples):
        img_tensor_dev = img_tensor.to(device)

        true_name = CIFAR100_CLASSES[true_lbl].replace('_', ' ')
        pred_b_name = CIFAR100_CLASSES[pred_b].replace('_', ' ')
        pred_h_name = CIFAR100_CLASSES[pred_h].replace('_', ' ')

        # 生成 CAM
        cam_b, _ = gradcam_b.generate_cam(img_tensor_dev)
        cam_h, _ = gradcam_h.generate_cam(img_tensor_dev)

        if cam_b is None or cam_h is None:
            continue

        img_orig, sup_b = apply_colormap_sci(cam_b, img_tensor)
        _, sup_h = apply_colormap_sci(cam_h, img_tensor)

        # === 第 1 列：原图 (GT) ===
        axes[idx, 0].imshow(img_orig, interpolation='lanczos')
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])
        # 边框设为黑色细线
        for spine in axes[idx, 0].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

        # 标签设置：加粗，居中对齐，增加 labelpad 防止重叠
        label_gt = f"GT: {true_name}"
        axes[idx, 0].set_xlabel(label_gt, fontsize=9, fontweight='bold', labelpad=6, ha='center')

        # === 第 2 列：Baseline (错误) ===
        axes[idx, 1].imshow(sup_b, interpolation='lanczos')
        axes[idx, 1].set_xticks([])
        axes[idx, 1].set_yticks([])

        label_b = f"Pred: {pred_b_name} $\\times$"
        # 深红色 (#B22222)，加粗
        axes[idx, 1].set_xlabel(label_b, fontsize=9, fontweight='bold', color='#B22222', labelpad=6, ha='center')

        # 红色边框强调
        for spine in axes[idx, 1].spines.values():
            spine.set_edgecolor('#B22222')
            spine.set_linewidth(1.2)

        # === 第 3 列：HFENet (正确) ===
        axes[idx, 2].imshow(sup_h, interpolation='lanczos')
        axes[idx, 2].set_xticks([])
        axes[idx, 2].set_yticks([])

        label_h = f"Pred: {pred_h_name} $\\checkmark$"
        # 深绿色 (#228B22)，加粗
        axes[idx, 2].set_xlabel(label_h, fontsize=9, fontweight='bold', color='#228B22', labelpad=6, ha='center')

        # 绿色边框强调
        for spine in axes[idx, 2].spines.values():
            spine.set_edgecolor('#228B22')
            spine.set_linewidth(1.2)

    # --- 4. 设置列标题 (Column Headers) ---
    # 不在每个 subplot 上设 title，而是在上方统一设一个大标题或者在每个列的最上方设标题
    # 这里采用在每个列的最上方 subplot 之上添加文本的方式，更灵活
    col_titles = ["Input Image", "Baseline (WRN)", "HFENet (Ours)"]

    # 获取第一个行的 axes 位置，用于在其上方放置标题
    # 我们需要计算每个列中心的 x 坐标
    positions = [axes[0, j].get_position() for j in range(3)]

    for j, title in enumerate(col_titles):
        # 计算该列的中心 x 坐标 (相对于 figure)
        x_center = (positions[j].x0 + positions[j].x1) / 2
        # y 坐标设在 top 下方一点点
        y_pos = top + 0.02

        fig.text(x_center, y_pos, title,
                 fontsize=11, fontweight='bold', ha='center', va='bottom',
                 color='black')

    # --- 5. 保存图像 ---
    out_png = os.path.join(save_dir, "fig3_gradcam_fixed_v2.png")
    out_pdf = os.path.join(save_dir, "fig3_gradcam_fixed_v2.pdf")

    # 关键：不使用 bbox_inches='tight'，因为我们已经手动计算了完美的 margins
    # 如果必须用 tight，可能会破坏我们精心计算的 bottom_margin
    plt.savefig(out_png, dpi=600, facecolor='white', edgecolor='none')
    plt.savefig(out_pdf, facecolor='white', edgecolor='none')

    print(f"✅ 图片已保存 (排版优化):\n   - {out_png}\n   - {out_pdf}")

    # 清理
    gradcam_b.remove_hooks()
    gradcam_h.remove_hooks()
    plt.close(fig)

def main():
    print("=" * 60)
    print("SCI Figure Generator: Grad-CAM Error Correction")
    print("=" * 60)

    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        # 如果是测试代码，可以在此处生成随机数据模拟，但正式运行请提供路径
        return

    # 1. Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    # 确保 ./data 目录下有 cifar-100-python 文件夹，或者设置 download=True
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    # 2. Load Models
    model_b, model_h = load_model(CHECKPOINT_PATH, device)
    if not model_b or not model_h:
        return

    # 3. Find Samples
    samples = find_error_samples(model_b, model_h, test_loader, device, num_samples=NUM_SAMPLES_TO_FIND)

    if not samples:
        print("\n❌ 未找到符合条件的样本。请检查模型性能或尝试其他 checkpoint。")
        return

    # 4. Plot
    plot_gradcam_comparison(samples, model_b, model_h, device, OUTPUT_DIR)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()