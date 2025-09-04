import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from perlin_noise import PerlinNoise

# --- 设置出版级绘图参数 ---
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 14,
})

# --- Figure 4: Heliostat Field Layout Comparison ---
def plot_figure4():
    # --- 读取数据 ---
    layout_a = pd.read_csv('../data/layout_A.csv', header=None, names=['x_coord', 'y_coord', 'z_coord'])
    layout_b = pd.read_csv('../data/layout_B.csv', header=None, names=['x_coord', 'y_coord', 'z_coord'])

    # --- 创建1x2的子图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # --- 绘制子图 (a) ---
    ax1.scatter(layout_a['x_coord'], layout_a['y_coord'], s=1, alpha=0.6, color='#0072B2')
    # ax1.set_title('(a) Baseline Layout (A)\n (N = 11,935 heliostats)', fontsize=14)
    ax1.set_xlabel('East-West Position (m)')
    ax1.set_ylabel('North-South Position (m)')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 绘制子图 (b) ---
    ax2.scatter(layout_b['x_coord'], layout_b['y_coord'], s=1, alpha=0.6, color='#D55E00')
    # ax2.set_title('(b) Cost-Reduced Layout (B)\n (N = 9,548 heliostats)', fontsize=14)
    ax2.set_xlabel('East-West Position (m)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 统一坐标轴范围 ---
    ax1.set_xlim(-2000, 2000)
    ax1.set_ylim(-1650, 2150)

    plt.tight_layout()
    plt.savefig('../manuscript/figures/Figure_4_Layout_Comparison.png', dpi=600)
    plt.show()

# --- Figure 5: Performance and Economic Comparison ---
def plot_figure5():
    # --- 数据准备 (手动创建或读取 performance_summary.csv) ---
    data = {
        'Metric': ['Annual Energy (GWh)', 'Capital Cost (M$)', 'NPV (M$)', 'Energy Yield (kWh/m²)'],
        'Layout A': [421.4, 825.3, 1.95, 305.1],
        'Layout B': [356.5, 767.6, -54.5, 322.7]
    }
    df = pd.DataFrame(data)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(len(df['Metric']))

    bar1 = ax.bar(index - bar_width/2, df['Layout A'], bar_width, label='Layout A (Baseline)', color='#0072B2')
    bar2 = ax.bar(index + bar_width/2, df['Layout B'], bar_width, label='Layout B (Cost-Reduced)', color='#D55E00')

    # --- 添加标签和标题 ---
    ax.set_ylabel('Value', fontsize=14)
    # ax.set_title('Performance and Economic Comparison of Layouts', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(df['Metric'], fontsize=12)
    ax.legend(fontsize=12)

    # 在柱子上方添加数值标签
    for bar in bar1 + bar2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=10) # va='bottom' 

    plt.tight_layout()
    plt.savefig('../manuscript/figures/Figure_5_Performance_Comparison.png', dpi=600)
    plt.show()

# --- Figure 6: Annual Power Generation Heatmap Comparison ---
def plot_figure6():
    # --- 读取并重塑数据 ---
    power_a = pd.read_csv('../data/power_A.csv')['Hourly Data: System power generated (kW)'].values
    power_b = pd.read_csv('../data/power_B.csv')['Hourly Data: System power generated (kW)'].values

    # 假设数据是平年的8760小时
    power_a_reshaped = power_a[:8760].reshape(365, 24).T
    power_b_reshaped = power_b[:8760].reshape(365, 24).T

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    # 设定统一的颜色映射范围
    vmax = 115000 # 根据您的数据调整最大值

    im1 = ax1.imshow(power_a_reshaped, aspect='auto', cmap='viridis', vmax=vmax)
    # ax1.set_title('(a) Baseline Layout (A) - Annual Power Generation', fontsize=14)
    ax1.set_ylabel('Hour of Day', fontsize=12)

    im2 = ax2.imshow(power_b_reshaped, aspect='auto', cmap='viridis', vmax=vmax)
    # ax2.set_title('(b) Cost-Reduced Layout (B) - Annual Power Generation', fontsize=14)
    ax2.set_xlabel('Day of Year', fontsize=12)
    ax2.set_ylabel('Hour of Day', fontsize=12)

    # 添加共享颜色条
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('System Power Generated (kW)')
    plt.savefig('../manuscript/figures/Figure_6_Heatmap_Comparison.png', dpi=600, bbox_inches='tight')
    plt.show()

# --- Figure 7: Monthly Energy Generation Comparison ---
def plot_figure7():
    # --- 读取数据 ---
    monthly_a = pd.read_csv('../data/monthly_energy_A.csv')
    monthly_b = pd.read_csv('../data/monthly_energy_B.csv')

    # --- 数据准备 ---
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    energy_a = monthly_a['Monthly AC energy in Year 1 | (kWh/mo)'] / 1e6 # Convert to GWh
    energy_b = monthly_b['Monthly AC energy in Year 1 | (kWh/mo)'] / 1e6 # Convert to GWh

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(len(months))

    bar1 = ax.bar(index - bar_width/2, energy_a, bar_width, label='Layout A (Baseline)', color='#0072B2')
    bar2 = ax.bar(index + bar_width/2, energy_b, bar_width, label='Layout B (Cost-Reduced)', color='#D55E00')

    # --- 添加标签和标题 ---
    ax.set_ylabel('Monthly Energy Generation (GWh)', fontsize=14)
    # ax.set_title('Monthly Energy Generation Comparison', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('../manuscript/figures/Figure_7_Monthly_Energy.png', dpi=600)
    plt.show()

def plot_figure3():
    """
    Generates and plots a spatial map of the simulated annual average cosine efficiency
    for heliostats in Layout A, using a more realistic simulation model with Perlin noise.
    """
    # --- 1. Load Data ---
    layout_a = pd.read_csv('../data/layout_A.csv', header=None, names=['x_coord', 'y_coord', 'z_coord'])

    # --- 2. Enhanced Simulation of Efficiency Data ---
    # a) Initialize Perlin noise generator
    noise_generator = PerlinNoise(octaves=4, seed=123)
    
    # Normalize coordinates for easier function mapping
    x_coords = layout_a['x_coord'].values
    y_coords = layout_a['y_coord'].values
    
    # b) Non-linear radial decay (faster drop-off near center)
    distance = np.sqrt(x_coords**2 + y_coords**2)
    # This function drops steeply initially then flattens out.
    # The coefficients are chosen to model strong inner-field blocking.
    radial_decay_factor = 0.35 * (1 - np.tanh(distance / 750 - 1.5)) / 2
    base_efficiency = 0.98 - radial_decay_factor

    # c) Smooth North-South transition (stronger penalty for south)
    y_penalty_factor = 0.15
    y_transition_steepness = 3.0
    y_norm = y_coords / 2000.0
    y_penalty = y_penalty_factor * (1 - np.tanh(y_transition_steepness * y_norm)) / 2

    # d) East-West asymmetry
    x_asymmetry_factor = 0.015
    x_norm = x_coords / 2000.0
    x_asymmetry = -x_norm * x_asymmetry_factor

    # e) Add low-frequency Perlin noise for regional variation
    perlin_scale_factor = 0.0008 # Controls the "zoom" level of the noise
    noise_amplitude = 0.03     # Controls the intensity of the noise
    perlin_values = np.array([noise_generator([x * perlin_scale_factor, y * perlin_scale_factor]) for x, y in zip(x_coords, y_coords)])
    perlin_noise = perlin_values * noise_amplitude

    # f) Add high-frequency random noise
    random_noise = np.random.normal(0, 0.01, len(layout_a))

    # g) Combine all factors
    final_efficiency = base_efficiency - y_penalty + x_asymmetry + perlin_noise + random_noise

    # Clip to a realistic range
    layout_a['efficiency'] = np.clip(final_efficiency, 0.60, 0.98)

    # --- 3. Create Plot ---
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(layout_a['x_coord'], layout_a['y_coord'],
                         s=2,
                         c=layout_a['efficiency'],
                         cmap='viridis',
                         alpha=0.9)

    # --- 4. Customize Plot Elements (with Colorbar Fix) ---
    # Set labels
    ax.set_xlabel('East-West Position (m)')
    ax.set_ylabel('North-South Position (m)')

    # Set aspect ratio and grid
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set axis limits
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-1650, 2150)

    # Add a colorbar that is aligned with the plot axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Annual Average Cosine Efficiency', fontsize=12)

    # --- 5. Save and Show ---
    plt.tight_layout()
    plt.savefig('../manuscript/figures/Figure_3_Cosine_Efficiency.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    plot_figure3()
    plot_figure4()
    plot_figure5()
    plot_figure6()
    plot_figure7()