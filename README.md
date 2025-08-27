# Dunhuang Heliostat Benchmark Dataset

A benchmark dataset of heliostat field layouts for the Dunhuang 100 MW solar tower plant, providing comprehensive data for concentrated solar power (CSP) research and optimization studies.

## Introduction

This dataset presents a comparative analysis of two heliostat field layouts for a concentrated solar power (CSP) plant based on the Shouhang Dunhuang 100 MW molten salt solar power tower facility in Dunhuang, China.

*   **Layout A (Baseline):** A standard, densely packed layout with 11,935 heliostats
*   **Layout B (Cost-Reduced):** An optimized layout with 9,548 heliostats, aiming for lower capital cost

The analysis reveals important trade-offs between capital cost reduction and energy yield, demonstrating potential "cost-optimization traps" in CSP plant design.

## Repository Structure

*   `code/`: Contains the Python script (`plot_generator.py`) to generate all figures from the data
*   `data/`: Contains the raw and processed data files:
    *   `layout_A.csv`, `layout_B.csv`: Heliostat coordinates (x, y, z) for both layouts
    *   `power_A.csv`, `power_B.csv`: Hourly power generation data (8760 hours)
    *   `monthly_energy_A.csv`, `monthly_energy_B.csv`: Monthly energy generation summaries
*   `manuscript/figures/`: Contains all figures and images:
    *   `Figure_1_Aerial_View.png`: Aerial view of the Dunhuang solar tower plant
    *   `Figure_2_DNI_Map.png`: DNI map showing plant location
    *   `Figure_3_Layout_Comparison.png`: Heliostat field layout comparison
    *   `Figure_4_Performance_Comparison.png`: Performance and economic metrics
    *   `Figure_5_Heatmap_Comparison.png`: Annual power generation heatmaps
    *   `Figure_6_Monthly_Energy.png`: Monthly energy generation comparison
*   `simulation_files/`: Contains SAM simulation files (`.sam`) for both layouts

## Reproducing the Figures

1.  **Prerequisites:**
    *   Python 3.x
    *   pandas
    *   matplotlib
    *   numpy

2.  **Run the script:**
    Navigate to the `code/` directory and run the script:
    ```bash
    python plot_generator.py
    ```
    The figures will be saved in the `manuscript/figures/` directory.