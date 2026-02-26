"""Generate all static figures for the 01-deep-learning-fundamentals notebooks.

Run from the img/ directory:
    python figure_gen.py

All figures are saved as PNG with 150 DPI (good balance between quality and file size).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = Path(__file__).parent
DPI = 150
STYLE = {"facecolor": "#FAFAFA"}


# ---------------------------------------------------------------------------
# 1. Framework trends (for 02_pytorch_basics)
# ---------------------------------------------------------------------------

def gen_framework_trends() -> None:
    years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    pytorch_share = np.array([35, 55, 70, 78, 85, 88, 89, 87, 85])
    tensorflow_share = np.array([55, 38, 22, 14, 9, 6, 5, 4, 4])
    jax_share = np.array([0, 0, 1, 2, 3, 4, 5, 7, 9])
    other_share = np.array([10, 7, 7, 6, 3, 2, 1, 2, 2])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=STYLE["facecolor"])
    ax.set_facecolor(STYLE["facecolor"])

    ax.plot(years, pytorch_share, color="#EE4C2C", linewidth=3, marker="o", label="PyTorch")
    ax.plot(years, tensorflow_share, color="#FF9900", linewidth=3, marker="s", label="TensorFlow")
    ax.plot(years, jax_share, color="#4285F4", linewidth=3, marker="^", label="JAX")
    ax.plot(years, other_share, color="#A0A0A0", linewidth=2, linestyle="--", label="Other (MXNet, Caffe, …)")

    ax.set_title(
        "Share of ML Research Paper Implementations by Framework\n(Historical Trend to Feb 2026)",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share of Implementations (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(2018, 2026)
    ax.set_xticks(years)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11, frameon=False)

    ax.axvline(x=2022.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(
        2022.6, 60, "Generative AI\n& LLM Boom", fontsize=10, color="gray",
        bbox={"facecolor": STYLE["facecolor"], "edgecolor": "none", "alpha": 0.8},
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "framework_trends_2026.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Perceptron diagram (for 01_intro)
# ---------------------------------------------------------------------------

def gen_perceptron_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=STYLE["facecolor"])
    ax.set_facecolor(STYLE["facecolor"])
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input nodes
    input_labels = ["$x_1$", "$x_2$", "$x_3$"]
    input_y = [3.5, 2.0, 0.5]
    input_x = 0.5

    for i, (label, y) in enumerate(zip(input_labels, input_y)):
        circle = plt.Circle((input_x, y), 0.35, color="#4A90D9", ec="black", lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(input_x, y, label, ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=4)

    # Summation node
    sum_x, sum_y = 3.5, 2.0
    circle = plt.Circle((sum_x, sum_y), 0.45, color="#F5A623", ec="black", lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(sum_x, sum_y, "$\\Sigma$", ha="center", va="center", fontsize=18, fontweight="bold", zorder=4)

    # Activation node
    act_x, act_y = 5.5, 2.0
    circle = plt.Circle((act_x, act_y), 0.45, color="#7ED321", ec="black", lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(act_x, act_y, "$f$", ha="center", va="center", fontsize=18, fontweight="bold", zorder=4)

    # Output node
    out_x, out_y = 7.0, 2.0
    circle = plt.Circle((out_x, out_y), 0.35, color="#D0021B", ec="black", lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(out_x, out_y, "$\\hat{y}$", ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=4)

    # Connections: inputs → summation
    weight_labels = ["$w_1$", "$w_2$", "$w_3$"]
    for i, (y, wlabel) in enumerate(zip(input_y, weight_labels)):
        ax.annotate(
            "", xy=(sum_x - 0.45, sum_y + (y - sum_y) * 0.3), xytext=(input_x + 0.35, y),
            arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.5},
        )
        mid_x = (input_x + 0.35 + sum_x - 0.45) / 2
        mid_y = (y + sum_y) / 2
        ax.text(mid_x - 0.15, mid_y + 0.2, wlabel, fontsize=11, color="#333", ha="center")

    # Bias arrow
    ax.annotate(
        "", xy=(sum_x, sum_y - 0.45), xytext=(sum_x, -0.8),
        arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.5},
    )
    ax.text(sum_x + 0.2, -0.8, "$b$ (bias)", fontsize=11, color="#333", va="center")

    # Summation → activation
    ax.annotate(
        "", xy=(act_x - 0.45, act_y), xytext=(sum_x + 0.45, sum_y),
        arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.5},
    )
    ax.text((sum_x + act_x) / 2, sum_y + 0.35, "$z = \\sum w_i x_i + b$", fontsize=10, ha="center", color="#555")

    # Activation → output
    ax.annotate(
        "", xy=(out_x - 0.35, out_y), xytext=(act_x + 0.45, act_y),
        arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.5},
    )

    # Title & labels
    ax.text(3.5, 4.3, "Perceptron", fontsize=16, fontweight="bold", ha="center")
    ax.text(input_x, -1.2, "Inputs", fontsize=11, ha="center", color="#4A90D9", fontweight="bold")
    ax.text(sum_x, -1.2, "Weighted\nsum", fontsize=10, ha="center", color="#F5A623", fontweight="bold")
    ax.text(act_x, -1.2, "Activation\nfunction", fontsize=10, ha="center", color="#7ED321", fontweight="bold")
    ax.text(out_x, -1.2, "Output", fontsize=11, ha="center", color="#D0021B", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "perceptron_diagram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. MLP architecture diagram (for 01_intro)
# ---------------------------------------------------------------------------

def gen_mlp_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=STYLE["facecolor"])
    ax.set_facecolor(STYLE["facecolor"])
    ax.axis("off")

    layer_sizes = [4, 6, 6, 3]
    layer_labels = ["Input\nlayer", "Hidden\nlayer 1", "Hidden\nlayer 2", "Output\nlayer"]
    layer_colors = ["#4A90D9", "#F5A623", "#F5A623", "#D0021B"]
    x_positions = [1, 3, 5, 7]
    max_neurons = max(layer_sizes)

    node_positions: list[list[tuple[float, float]]] = []

    for layer_idx, (n, x) in enumerate(zip(layer_sizes, x_positions)):
        y_start = (max_neurons - n) / 2
        positions = [(x, y_start + i * 1.0) for i in range(n)]
        node_positions.append(positions)

        for px, py in positions:
            circle = plt.Circle((px, py), 0.25, color=layer_colors[layer_idx], ec="black", lw=1, zorder=3)
            ax.add_patch(circle)

        ax.text(x, -1.2, layer_labels[layer_idx], fontsize=10, ha="center", fontweight="bold",
                color=layer_colors[layer_idx])

    # Draw connections between consecutive layers
    for l in range(len(layer_sizes) - 1):
        for x1, y1 in node_positions[l]:
            for x2, y2 in node_positions[l + 1]:
                ax.plot([x1 + 0.25, x2 - 0.25], [y1, y2], color="#CCCCCC", lw=0.5, zorder=1)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-2, max_neurons + 0.5)
    ax.set_aspect("equal")
    ax.set_title("Multilayer Perceptron (MLP)", fontsize=16, fontweight="bold", pad=20)

    # Feature interpretation annotations
    annotations = [
        (1, max_neurons + 0.2, "Raw\nfeatures", "#4A90D9"),
        (4, max_neurons + 0.2, "Learned intermediate\nrepresentations", "#F5A623"),
        (7, max_neurons + 0.2, "Final\nprediction", "#D0021B"),
    ]
    for ax_x, ay, text, color in annotations:
        ax.text(ax_x, ay, text, fontsize=9, ha="center", color=color, style="italic")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mlp_diagram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Activation functions comparison (for 01_intro)
# ---------------------------------------------------------------------------

def gen_activation_functions() -> None:
    x = np.linspace(-5, 5, 300)

    activations = [
        ("Sigmoid", 1 / (1 + np.exp(-x)), "(0, 1)"),
        ("Tanh", np.tanh(x), "(-1, 1)"),
        ("ReLU", np.maximum(0, x), "[0, ∞)"),
        ("Leaky ReLU (α=0.1)", np.where(x > 0, x, 0.1 * x), "(-∞, ∞)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), facecolor=STYLE["facecolor"])

    for ax, (name, y, range_str) in zip(axes, activations):
        ax.set_facecolor(STYLE["facecolor"])
        ax.plot(x, y, color="#2196F3", linewidth=2.5)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("x", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.text(
            0.95, 0.05, f"Range: {range_str}", transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", color="#666",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    fig.suptitle("Common Activation Functions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "activation_functions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Gradient descent on simple surface (for 01_intro)
# ---------------------------------------------------------------------------

def gen_gradient_descent_surface() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=STYLE["facecolor"])

    # Left: 2D contour with gradient descent path
    ax = axes[0]
    ax.set_facecolor(STYLE["facecolor"])

    w1 = np.linspace(-3, 3, 200)
    w2 = np.linspace(-3, 3, 200)
    W1, W2 = np.meshgrid(w1, w2)
    # Simple bowl-shaped loss: L(w1, w2) = w1² + 2*w2²
    Z = W1**2 + 2 * W2**2

    ax.contour(W1, W2, Z, levels=15, cmap="viridis", alpha=0.6)
    contour_filled = ax.contourf(W1, W2, Z, levels=15, cmap="viridis", alpha=0.3)

    # Simulate gradient descent path
    lr = 0.15
    path_w1, path_w2 = [2.5], [2.5]
    for _ in range(20):
        grad_w1 = 2 * path_w1[-1]
        grad_w2 = 4 * path_w2[-1]
        path_w1.append(path_w1[-1] - lr * grad_w1)
        path_w2.append(path_w2[-1] - lr * grad_w2)

    ax.plot(path_w1, path_w2, "o-", color="#EE4C2C", markersize=4, linewidth=1.5, label="GD path")
    ax.plot(path_w1[0], path_w2[0], "o", color="#EE4C2C", markersize=10, zorder=5)
    ax.plot(0, 0, "*", color="#FFD700", markersize=15, zorder=5, markeredgecolor="black", label="Minimum")
    ax.annotate("Start", (path_w1[0], path_w2[0]), textcoords="offset points", xytext=(10, 10), fontsize=10)
    ax.set_xlabel("$w_1$", fontsize=12)
    ax.set_ylabel("$w_2$", fontsize=12)
    ax.set_title("Gradient descent: 2D view\n\"Walk downhill\"", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # Right: 3D surface
    ax3d = fig.add_subplot(122, projection="3d", facecolor=STYLE["facecolor"])
    axes[1].remove()
    W1_coarse = np.linspace(-3, 3, 50)
    W2_coarse = np.linspace(-3, 3, 50)
    W1c, W2c = np.meshgrid(W1_coarse, W2_coarse)
    Zc = W1c**2 + 2 * W2c**2

    ax3d.plot_surface(W1c, W2c, Zc, cmap="viridis", alpha=0.6, edgecolor="none")
    path_z = [w**2 + 2 * v**2 for w, v in zip(path_w1, path_w2)]
    ax3d.plot(path_w1, path_w2, path_z, "o-", color="#EE4C2C", markersize=3, linewidth=1.5, zorder=5)
    ax3d.set_xlabel("$w_1$")
    ax3d.set_ylabel("$w_2$")
    ax3d.set_zlabel("Loss")
    ax3d.set_title("Loss surface", fontsize=12, fontweight="bold")
    ax3d.view_init(elev=35, azim=225)

    fig.suptitle(
        "Gradient Descent: Iteratively Adjust Parameters to Minimize Loss",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gradient_descent_surface.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. XOR / AND / OR scatter (for 01_intro)
# ---------------------------------------------------------------------------

def gen_xor_scatter() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=STYLE["facecolor"])

    problems = [
        ("AND", [0, 0, 0, 1], True),
        ("OR", [0, 1, 1, 1], True),
        ("XOR", [0, 1, 1, 0], False),
    ]
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for ax, (name, labels, separable) in zip(axes, problems):
        ax.set_facecolor(STYLE["facecolor"])
        labels_arr = np.array(labels)

        # Plot points
        for cls, color, marker in [(0, "#4A90D9", "o"), (1, "#D0021B", "s")]:
            mask = labels_arr == cls
            ax.scatter(inputs[mask, 0], inputs[mask, 1], c=color, marker=marker,
                       s=120, zorder=3, edgecolors="black", linewidths=1, label=f"Class {cls}")

        # Draw decision boundary if separable
        if separable:
            if name == "AND":
                x_line = np.linspace(-0.5, 1.5, 100)
                y_line = 1.5 - x_line  # x1 + x2 = 1.5
                ax.plot(x_line, y_line, "--", color="#7ED321", linewidth=2, label="Decision boundary")
                ax.fill_between(x_line, y_line, 1.5, alpha=0.1, color="#D0021B")
                ax.fill_between(x_line, -0.5, y_line, alpha=0.1, color="#4A90D9")
            elif name == "OR":
                x_line = np.linspace(-0.5, 1.5, 100)
                y_line = 0.5 - x_line  # x1 + x2 = 0.5
                ax.plot(x_line, y_line, "--", color="#7ED321", linewidth=2, label="Decision boundary")
                ax.fill_between(x_line, y_line, 1.5, alpha=0.1, color="#D0021B")
                ax.fill_between(x_line, -0.5, y_line, alpha=0.1, color="#4A90D9")
        else:
            # XOR: show that no single line works
            ax.text(0.5, -0.35, "No single line can separate these!", fontsize=9, ha="center",
                    color="#999", style="italic")

        status = "✓ Linearly separable" if separable else "✗ NOT linearly separable"
        status_color = "#7ED321" if separable else "#D0021B"
        ax.set_title(f"{name}\n{status}", fontsize=12, fontweight="bold", color=status_color)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("$x_1$", fontsize=11)
        ax.set_ylabel("$x_2$", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    fig.suptitle("Logical Gates: Linear Separability", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "xor_and_or_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Feature hierarchy diagram (for 01_intro — MLP section)
# ---------------------------------------------------------------------------

def gen_feature_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(12, 3), facecolor=STYLE["facecolor"])
    ax.set_facecolor(STYLE["facecolor"])
    ax.axis("off")

    stages = [
        ("Pixels", "#4A90D9", "Raw input\n(784 values)"),
        ("Edges &\nTextures", "#F5A623", "Layer 1\ndetects"),
        ("Shapes &\nParts", "#7ED321", "Layer 2\ncombines"),
        ("Objects", "#D0021B", "Output\nclassifies"),
    ]

    for i, (label, color, desc) in enumerate(stages):
        x = i * 3
        rect = mpatches.FancyBboxPatch(
            (x, 0.5), 2, 1.5, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + 1, 1.25, label, fontsize=12, fontweight="bold", ha="center", va="center", color="white")
        ax.text(x + 1, -0.1, desc, fontsize=9, ha="center", va="top", color="#555")

        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x + 2.6, 1.25), xytext=(x + 2.1, 1.25),
                arrowprops={"arrowstyle": "->", "color": "#888", "lw": 2},
            )

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 2.8)
    ax.set_title(
        "How hidden layers build features (image classification example)",
        fontsize=13, fontweight="bold", pad=10,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_hierarchy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main: generate all figures
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generators = [
        ("framework_trends_2026.png", gen_framework_trends),
        ("perceptron_diagram.png", gen_perceptron_diagram),
        ("mlp_diagram.png", gen_mlp_diagram),
        ("activation_functions.png", gen_activation_functions),
        ("gradient_descent_surface.png", gen_gradient_descent_surface),
        ("xor_and_or_scatter.png", gen_xor_scatter),
        ("feature_hierarchy.png", gen_feature_hierarchy),
    ]
    for name, func in generators:
        print(f"Generating {name}...")
        func()
        print(f"  ✓ Saved to {OUTPUT_DIR / name}")

    print(f"\nDone! Generated {len(generators)} figures in {OUTPUT_DIR}")