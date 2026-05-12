import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure with better styling - more compact
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(8, 18), facecolor='white')
ax.set_xlim(0, 8)
ax.set_ylim(0, 32)
ax.axis('off')

# Professional color scheme
colors = {
    'input': '#34495e',      # Dark blue-grey
    'encoder': '#3498db',    # Professional blue
    'latent': '#e74c3c',     # Strong red for emphasis
    'decoder': '#27ae60',    # Professional green
    'text': '#2c3e50'        # Dark text
}

# Title section
ax.text(4, 30.5, 'LLE Bi-Exponential Autoencoder',
        ha='center', fontsize=24, fontweight='bold', color=colors['text'])
ax.text(4, 29.7, 'Neural Network Architecture',
        ha='center', fontsize=16, color='#7f8c8d', style='italic')

# Draw a subtle horizontal divider
ax.plot([0.3, 7.7], [29.2, 29.2], '-', color='#bdc3c7', linewidth=1, alpha=0.5)

y_pos = 28
layer_height = 0.7
spacing = 0.4

def get_width_from_dims(dim_value):
    """Map dimensionality to visual width (logarithmic scale for better visualization)"""
    # Handle string labels (e.g., "2×256", "32×32")
    if isinstance(dim_value, str):
        if 'x' in dim_value.lower() or '×' in dim_value:
            return 6.5  # Wide for multi-dimensional
        return 5.0

    # Handle numeric values
    if dim_value <= 3:
        return 2.0  # Minimum width for latent
    elif dim_value <= 64:
        return 3.5
    elif dim_value <= 1024:
        return 5.0
    else:
        return 6.5

def draw_layer(y, dim_value, height, label, details, color, is_block=False):
    """Draw a layer with professional styling"""
    width = get_width_from_dims(dim_value)
    x = 4 - width/2

    # Draw shadow for depth
    shadow = Rectangle((x + 0.08, y - 0.08), width, height,
                       facecolor='black', alpha=0.15, zorder=1)
    ax.add_patch(shadow)

    # Draw main rectangle (square edges)
    rect = Rectangle((x, y), width, height,
                     facecolor=color, edgecolor='#2c3e50',
                     linewidth=2, alpha=0.85, zorder=2)
    ax.add_patch(rect)

    # Add text with larger font
    ax.text(4, y + height * 0.65, label,
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='white', zorder=3)

    if details:
        ax.text(4, y + height * 0.28, details,
                ha='center', va='center', fontsize=10,
                color='white', alpha=0.95, zorder=3)

    # Add dimension annotation on the right
    dim_text = f"{dim_value}" if isinstance(dim_value, int) else dim_value
    ax.text(7.4, y + height/2, dim_text,
            ha='left', va='center', fontsize=11,
            color=colors['text'], style='italic')

    return width

def draw_arrow(y_start, y_end, width_start, width_end):
    """Draw a tapered arrow showing dimensionality change"""
    x_left_start = 4 - width_start/2
    x_right_start = 4 + width_start/2
    x_left_end = 4 - width_end/2
    x_right_end = 4 + width_end/2

    # Draw arrow shaft as a polygon for tapering effect
    if abs(width_start - width_end) > 0.1:
        # Tapered connector
        polygon = plt.Polygon([
            (x_left_start, y_start),
            (x_right_start, y_start),
            (x_right_end, y_end),
            (x_left_end, y_end)
        ], facecolor='#95a5a6', alpha=0.3, zorder=0)
        ax.add_patch(polygon)

    # Center arrow
    arrow = FancyArrowPatch((4, y_start), (4, y_end),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='#34495e', alpha=0.7, zorder=1)
    ax.add_patch(arrow)

# ===== BUILD THE ARCHITECTURE =====

# Input
w_prev = draw_layer(y_pos, "2×256", layer_height, 'Input Signals',
                    'Decay & IRF signals', colors['input'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(32*32))

# Embedding
y_pos -= layer_height
w_prev = draw_layer(y_pos, "32×32", layer_height, 'Conv1d Embeddings',
                    '1→16 channels each, k=8, s=8 → concat', colors['encoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, w_prev)

# Encoder blocks
y_pos -= layer_height * 1.8
w_prev = draw_layer(y_pos, "32×32", layer_height * 1.8,
                    '8 × Encoder Blocks',
                    'Residual + DepthConv(k=9) + PointConv + GELU + BN',
                    colors['encoder'], is_block=True)
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(1024))

# Flatten
y_pos -= layer_height
w_prev = draw_layer(y_pos, 1024, layer_height, 'Flatten',
                    '32 × 32 = 1024 features', colors['encoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(64))

# FC1
y_pos -= layer_height
w_prev = draw_layer(y_pos, 64, layer_height, 'Linear + ReLU',
                    '1024 → 64', colors['encoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(3))

# LATENT SPACE (BOTTLENECK)
y_pos -= layer_height * 1.5
w_prev = draw_layer(y_pos, 3, layer_height * 1.5, 'LATENT SPACE',
                    r'$\tau_1$, $\tau_2$, f', colors['latent'])

y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, get_width_from_dims(3), get_width_from_dims(64))

# Decoder FC1
y_pos -= layer_height
w_prev = draw_layer(y_pos, 64, layer_height, 'Linear + ReLU',
                    '3 → 64', colors['decoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(1024))

# Decoder FC2
y_pos -= layer_height
w_prev = draw_layer(y_pos, 1024, layer_height, 'Linear + ReLU',
                    '64 → 1024', colors['decoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(32*32))

# Reshape
y_pos -= layer_height
w_prev = draw_layer(y_pos, "32×32", layer_height, 'Reshape',
                    '1024 → 32 × 32', colors['decoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, w_prev)

# Decoder blocks
y_pos -= layer_height * 1.8
w_prev = draw_layer(y_pos, "32×32", layer_height * 1.8,
                    '8 × Decoder Blocks',
                    'Residual + DepthConv(k=9) + PointConv + GELU + BN',
                    colors['decoder'], is_block=True)
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(16*256))

# Upsample
y_pos -= layer_height
w_prev = draw_layer(y_pos, "16×256", layer_height, 'ConvTranspose1d',
                    '32→16 channels, k=8, s=8', colors['decoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, get_width_from_dims(256))

# Output Conv
y_pos -= layer_height
w_prev = draw_layer(y_pos, 256, layer_height, 'Output Conv1d',
                    '16→1 channel, k=1', colors['decoder'])
y_pos -= spacing
draw_arrow(y_pos + spacing, y_pos, w_prev, w_prev)

# Final Output
y_pos -= layer_height
draw_layer(y_pos, 256, layer_height, 'Reconstructed Signal',
           '1 × 256 + Tanh', colors['input'])

# ===== ANNOTATIONS =====

# Dimension label on right (positioned above the first layer)
ax.text(7.4, 28.5, 'Dimensions', fontsize=11, fontweight='bold',
        ha='left', va='bottom', color=colors['text'])

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors['input'], edgecolor='#2c3e50',
                   linewidth=2, label='Input/Output'),
    mpatches.Patch(facecolor=colors['encoder'], edgecolor='#2c3e50',
                   linewidth=2, label='Encoder'),
    mpatches.Patch(facecolor=colors['latent'], edgecolor='#2c3e50',
                   linewidth=2, label='Latent Space'),
    mpatches.Patch(facecolor=colors['decoder'], edgecolor='#2c3e50',
                   linewidth=2, label='Decoder')
]

legend = ax.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=4, fontsize=11, framealpha=0.95,
                   edgecolor='#2c3e50', fancybox=True)

# Add metadata box
info_text = (
    'Model: LLE Bi-Exponential Autoencoder\n'
    'Encoder: 8 ConvLayers -> Flatten -> 2 FC\n'
    'Decoder: 2 FC -> Reshape -> 8 ConvLayers -> Upsample\n'
    r'Latent: 3D ($\tau_1$, $\tau_2$, f)'
)
ax.text(0.4, 1.5, info_text, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                 edgecolor='#95a5a6', linewidth=1, alpha=0.9),
        family='monospace', color=colors['text'])

plt.tight_layout()
plt.savefig('C:/Users/mcg11923/Thesis/FPFLI/encodertest11_21/training_logs_fully_flatrandom_n0100_20260219_235039/model_architecture_visualization.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Professional visualization saved to: model_architecture_visualization.png")
plt.close()
