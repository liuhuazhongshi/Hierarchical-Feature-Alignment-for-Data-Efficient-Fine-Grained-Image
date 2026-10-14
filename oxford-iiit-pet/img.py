import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

# -------------------------
# 1. CBA Attention Visualization
# -------------------------

# Example CIFAR image (32x32), upsampled
img = np.random.rand(32, 32, 3)  # replace with actual image
img_up = np.kron(img, np.ones((8, 8, 1)))  # upsample to 256x256

# Example feature maps (32x32)
feat_before = np.random.rand(32, 32)
attention_map = np.random.rand(32, 32)
feat_after = feat_before * attention_map

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# (a) Input image
axes[0].imshow(img_up)
axes[0].set_title('(a)', loc='left', fontsize=12)
axes[0].axis('off')

# (b) Feature before CBA
axes[1].imshow(feat_before, cmap='jet', alpha=0.6)
axes[1].set_title('(b)', loc='left', fontsize=12)
axes[1].axis('off')

# (c) CBA attention
axes[2].imshow(attention_map, cmap='jet', alpha=0.6)
axes[2].set_title('(c)', loc='left', fontsize=12)
axes[2].axis('off')

# (d) Feature after CBA
axes[3].imshow(feat_after, cmap='jet', alpha=0.6)
axes[3].set_title('(d)', loc='left', fontsize=12)
axes[3].axis('off')

plt.tight_layout()
plt.savefig('CBA_Attention.png', dpi=300)
plt.show()

# -------------------------
# 2. t-SNE Feature Distribution Visualization
# -------------------------

# Simulate 4 models, 10 classes, 50 samples per class
num_classes = 10
samples_per_class = 50
models = ['WRN-16-8', 'CBAM-WRN', 'XNet', 'HFENet']

# Random high-dimensional features
features_dict = {m: np.random.rand(num_classes * samples_per_class, 128) for m in models}
labels = np.repeat(np.arange(num_classes), samples_per_class)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

colors = cm.tab10(np.linspace(0, 1, num_classes))

for i, model in enumerate(models):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
    X_2d = tsne.fit_transform(features_dict[model])

    for c in range(num_classes):
        axes[i].scatter(
            X_2d[labels == c, 0],
            X_2d[labels == c, 1],
            s=8,
            color=colors[c],
            alpha=0.7,
            label=f'Class {c}' if i == 0 else ""
        )
    axes[i].set_title(f'({chr(97 + i)}) {model}' + (' (ours)' if model == 'HFENet' else ''), loc='left', fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('tSNE_Features.png', dpi=300)
plt.show()
