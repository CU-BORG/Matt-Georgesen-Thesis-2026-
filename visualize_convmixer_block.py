#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ConvMixer Block Diagram - Vertical Layout
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6, 10))
ax.set_xlim(0, 6)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors - all boxes white with black outlines
c_data = 'white'
c_depthwise = 'white'
c_pointwise = 'white'
c_residual = 'white'

box_width = 2.5
box_height = 0.8
x_center = 3

# Helper function to draw a box
def draw_box(x, y, width, height, color, label, sublabel=''):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=12, fontweight='bold')
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                fontsize=9)

# Helper function to draw arrow that connects blocks
def draw_arrow(y1, y2, x, label=''):
    # Adjust arrow to start from bottom of upper box and end at top of lower box
    arrow = FancyArrowPatch((x, y1), (x, y2),
                           arrowstyle='->,head_width=0.3,head_length=0.2',
                           color='black', linewidth=2)
    ax.add_artist(arrow)
    if label:
        ax.text(x + 0.3, (y1 + y2)/2, label, ha='left', va='center', fontsize=8)

# Title
ax.text(3, 9.5, 'ConvMixer Block', ha='center', fontsize=14, fontweight='bold')

# Flow
y_pos = 8.5
arrow_gap = 0.6  # Gap for arrow between blocks

# Input
draw_box(x_center, y_pos, box_width, box_height, c_data, 'Input', '[B, 32, 32]')
input_y = y_pos

# Arrow from Input to Depthwise
y_start = input_y - box_height/2
y_pos = y_start - arrow_gap
draw_arrow(y_start, y_pos, x_center)

# Depthwise Conv
y_pos -= box_height/2
draw_box(x_center, y_pos, box_width, box_height, c_depthwise, 'Depthwise Conv', 'k=9, groups=32')

# Arrow to GELU+BN
y_start = y_pos - box_height/2
y_pos = y_start - arrow_gap
draw_arrow(y_start, y_pos, x_center)

# GELU + BN
y_pos -= 0.3
draw_box(x_center, y_pos, 1.8, 0.6, 'white', 'GELU + BN')

# Arrow to Residual Add
y_start = y_pos - 0.3
y_pos = y_start - arrow_gap
draw_arrow(y_start, y_pos, x_center)

# Residual Add (circle)
y_pos -= 0.3
circle = plt.Circle((x_center, y_pos), 0.3, facecolor=c_residual, edgecolor='black', linewidth=2)
ax.add_patch(circle)
ax.text(x_center, y_pos, '+', ha='center', va='center', fontsize=18, fontweight='bold')
add_y = y_pos

# Residual connection (from input to add) - changed from "Skip" to "Residual"
skip_x = x_center + 2
# Horizontal from input
ax.plot([x_center + box_width/2, skip_x], [input_y, input_y],
        color='black', linewidth=2, linestyle='--')
# Vertical down
ax.plot([skip_x, skip_x], [input_y, add_y],
        color='black', linewidth=2, linestyle='--')
# Arrow to add
arrow_skip = FancyArrowPatch((skip_x, add_y), (x_center + 0.3, add_y),
                            arrowstyle='->,head_width=0.3,head_length=0.2',
                            color='black', linewidth=2, linestyle='--')
ax.add_artist(arrow_skip)
ax.text(skip_x + 0.2, (input_y + add_y)/2, 'Residual', rotation=-90,
        ha='center', va='center', fontsize=9, color='black', fontweight='bold')

# Arrow to Pointwise
y_start = y_pos - 0.3
y_pos = y_start - arrow_gap
draw_arrow(y_start, y_pos, x_center)

# Pointwise Conv
y_pos -= box_height/2
draw_box(x_center, y_pos, box_width, box_height, c_pointwise, 'Pointwise Conv', 'k=1')

# Arrow to GELU+BN
y_start = y_pos - box_height/2
y_pos = y_start - arrow_gap
draw_arrow(y_start, y_pos, x_center)

# GELU + BN
y_pos -= 0.3
draw_box(x_center, y_pos, 1.8, 0.6, 'white', 'GELU + BN')

# Arrow to Output
y_start = y_pos - 0.3
y_pos = y_start - arrow_gap
draw_arrow(y_start, y_pos, x_center)

# Output
y_pos -= box_height/2
draw_box(x_center, y_pos, box_width, box_height, c_data, 'Output', '[B, 32, 32]')

# Legend removed for cleaner look

plt.tight_layout()
plt.savefig('convmixer_block_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: convmixer_block_diagram.png")
# Don't show in interactive mode
# plt.show()
