import argparse

def parse_args():
    parser = argparse.ArgumentParser("Hierarchical Disentangled CIFAR-100 消融实验")
    parser.add_argument("--epoch", type=int, default=400, help="Total number of epochs to train")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--drop", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lambda_kl", type=float, default=0.2, help="层次KL散度损失的权重")
    parser.add_argument("--lambda_orth", type=float, default=5e-4, help="正交损失的权重")
    parser.add_argument("--lambda_fine", type=float, default=2.0, help="细粒度分类损失的权重")
    parser.add_argument("--warm", type=int, default=10, help="线性预热的epoch数")
    parser.add_argument("--no_reg", action="store_true", help="不使用正则化损失")
    parser.add_argument("--save_dir", type=str, default="ckpt", help="保存检查点的目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点文件路径")
    parser.add_argument("--num_workers", type=int, default=6, help="数据加载器的工作进程数")
    parser.add_argument("--use_cutmix", type=lambda x: x.lower() == 'true', default=True, help="是否使用CutMix数据增强")
    parser.add_argument("--strong_aug", action="store_true", default=True, help="是否使用更强的数据增强")
    
    # 消融实验参数
    parser.add_argument("--depth", type=int, default=16, help="WideResNet depth")
    parser.add_argument("--widen", type=int, default=8, help="WideResNet widen factor")
    parser.add_argument("--use_coarse_branch", type=bool, default=False, help="是否使用粗粒度分支")
    parser.add_argument("--use_cross_attention", type=bool, default=False, help="是否使用跨分支注意力")
    parser.add_argument("--use_kl", type=bool, default=False, help="是否使用层次KL散度")
    parser.add_argument("--use_dynamic_kl", type=bool, default=False, help="是否使用动态KL权重")
    parser.add_argument("--kl_warm_epochs", type=int, default=20, help="KL损失权重的warm-up epoch数")
    parser.add_argument("--kl_strategy", type=str, default="linear", choices=["linear", "cosine", "exponential", "logistic"], help="KL损失权重的动态调整策略")
    parser.add_argument("--kl_power", type=float, default=1.0, help="KL损失权重的动态调整指数（仅用于exponential策略）")
    parser.add_argument("--use_sam", type=bool, default=False, help="是否使用SAM优化器")
    # 注意力机制消融参数
    parser.add_argument("--attention_type", type=str, default="cross_branch", choices=["none", "se", "cbam", "cross_branch"], help="注意力类型（none: 无注意力, se: SE通道注意力, cbam: CBAM注意力, cross_branch: 跨分支注意力）")
    # 内存优化参数
    parser.add_argument("--image_size", type=int, default=224, help="输入图像尺寸（默认112，原224）")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--fp16", type=lambda x: x.lower() == 'true', default=False, help="是否使用混合精度训练")
    

    
    return parser.parse_args()