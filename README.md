
# RealTraffic: High-Fidelity Peak-Period Traffic Simulation via Congestion Maps

**RealTraffic** is a framework for building **high-fidelity peak-period traffic simulations** for **arbitrary urban regions** using **ubiquitous congestion maps** (e.g., Azure Maps / Amap), without requiring dense roadside sensors.

It automatically:
- downloads road networks from OpenStreetMap,
- collects multi-day congestion maps from map services,
- extracts stable peak-hour traffic patterns via **DTW alignment**,
- calibrates demand through a **bi-level calibration** strategy,
- runs **high-quality microscopic simulation** in SUMO,
- and supports downstream tasks such as **traffic signal control optimization**.

📌 Paper: *RealTraffic: High-Fidelity Peak-Period Traffic Simulation for Arbitrary Regions via Congestion Maps*  
🔗 Code: this repository

---

## 🔥 What You Can Do with RealTraffic

Given a region (latitude & longitude + bounding box), you can:

✅ Build a SUMO simulation automatically  
✅ Calibrate OD demand using congestion maps  
✅ Reproduce realistic peak-period congestion  
✅ Train and evaluate traffic optimization policies:
- Traffic light control (TLC)
- TLC + lane-changing coordination

---

## 🧭 User Workflow (4 Steps)

### Step 1 — Setup Environment
Install Python + SUMO and dependencies.

### Step 2 — Input Coordinates
Provide a region center or bounding box (lat/lon).
RealTraffic will automatically:
- fetch road topology (OSM),
- collect congestion maps (Azure Maps / Amap),
- construct simulation environment.

### Step 3 — Choose a Task
Choose one of:
- **Simulation**: build + calibrate high-fidelity traffic flows
- **Optimization**: train traffic control policies using calibrated simulation

### Step 4 — Run
Run a single command to obtain:
- calibrated simulation outputs
- metrics (Recall/Precision/F1, speed error, travel time error)
- visualization figures

---

## 📷 Visual Overview

### RealTraffic Pipeline
<p align="center">
  <img src="assets/architecture.png" width="900"/>
</p>

> RealTraffic: DTW-based pattern extraction + bi-level calibration + simulation + downstream optimization.

### Sim-to-Real Motivation
<p align="center">
  <img src="assets/motivation.png" width="700"/>
</p>

### Example: Congestion Map Alignment (DTW)
<p align="center">
  <img src="assets/dtw_alignment.png" width="900"/>
</p>

---

## ⚙️ Installation

### 1) Requirements

- **Python**: 3.10+ (recommended 3.11)
- **SUMO**: required (uses `traci` / `sumolib`)
- **OSM tools**: automatically handled by the pipeline

#### Install SUMO
- Linux: `sudo apt install sumo sumo-tools sumo-doc`
- macOS: `brew install sumo`
- Windows: download from SUMO official website

Then set:

```bash
export SUMO_HOME=/path/to/sumo
````

If not set, you may see:

```
Please set SUMO_HOME environment variable
```

---

### 2) Install Python Dependencies

```bash
pip install -r requirements.txt
```

> Note: For RL optimization tasks, you may need PyTorch.
> Install it based on your hardware from the official PyTorch page:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 🚀 Quick Start

The main entry is:

```bash
python run.py
```

---

## 🗺️ Run on Any Region (Lat/Lon Input)

### Option A: Run with a Config File (Recommended)

```bash
python run.py -t simulation -c configs/simu_custom.yml
```

Example config fields:

```yaml
region:
  name: "my_region"
  center_lat: 30.5928
  center_lon: 114.3055
  radius_km: 3

data:
  provider: "amap"     # "azure_maps" or "amap"
  days: 5
  peak_window: ["06:00", "10:00"]
  sample_interval_min: 20

calibration:
  enable_dtw: true
  max_iters: 10
  epsilon: 0.1
```

---

### Option B: Run with CLI Arguments (If Supported)

Some versions support passing coordinates directly:

```bash
python run.py -t simulation --lat 30.5928 --lon 114.3055 --radius 3
```

---

## 🧪 Task 1: High-Fidelity Simulation (Calibration)

Run:

```bash
python run.py -t simulation -c configs/simu_Manha.yml
```

Outputs include:

* calibrated OD trips
* simulated congestion states
* evaluation metrics (F1 / Recall / Precision)
* visualizations

---

## 🎯 Task 2: Traffic Optimization

### 2.1 Traffic Light Control (TLC)

```bash
python run.py -t traffic_light_optimization -c configs/opti_task_TLC_Manha_dqn.yml
```

### 2.2 TLC + Lane Coordination

```bash
python run.py -t traffic_light_optimization -c configs/opti_task_TLC_Lane.yml
```

---

## 📊 Outputs

After running optimization, results are saved to:

```
optimization_run_YYYYMMDD_HHMMSS/
```

Common files:

* `optimization_summary.json`
* training curves
* evaluation logs
* exported simulation figures

---

## 📁 Project Structure

```
.
├── run.py                          # main entry
├── configs/                         # configuration files
├── requirements.txt
├── realtraffic/                     # RealTraffic core modules
│   ├── data/                        # congestion map ingestion
│   ├── dtw/                         # DTW barycenter alignment
│   ├── env/                         # OSM -> SUMO environment build
│   ├── calibration/                 # bi-level calibration
│   ├── simulation/                  # SUMO wrapper + traci
│   └── optimization/                # RL / baselines
├── assets/                          # README figures
└── outputs/
```

---

## 🌍 Supported Data Providers

RealTraffic supports congestion map ingestion from:

* **Azure Maps** (US cities)
* **Amap (AutoNavi)** (China)

You need an API key.

---

## 🔑 API Key Setup

Create a file:

```
configs/api_keys.yaml
```

Example:

```yaml
azure_maps:
  subscription_key: "YOUR_AZURE_API_KEY"
  api_version: "1.0"
  style: "relative"
  zoom_level: 12

amap:
  api_key: "YOUR_AMAP_API_KEY"
  extensions: "all"
```

---

## 🧠 Method Summary (Paper Highlights)

### 1) DTW-based Multi-day Pattern Extraction

Peak congestion shifts by 10–30 minutes across days.
We use **DTW Barycenter Averaging (DBA)** to align multi-day sequences and extract stable congestion patterns.

### 2) Bi-Level Calibration

* **Level 1**: binary search a global scaling factor for OD volume
* **Level 2**: sign-gradient trip adjustment to match road-level congestion states

---

## 🧾 Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{realtraffic2026,
  title={RealTraffic: High-Fidelity Peak-Period Traffic Simulation for Arbitrary Regions via Congestion Maps},
  author={Liu, Tengfei and Li, Guanzhen and Wang, Leye},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

---

## 📬 Contact

* Tengfei Liu: [liutengfei2024@stu.pku.edu.cn](mailto:liutengfei2024@stu.pku.edu.cn)
* Guanzhen Li: [gzli25@stu.pku.edu.cn](mailto:gzli25@stu.pku.edu.cn)
* Leye Wang (Corresponding): [leyewang@pku.edu.cn](mailto:leyewang@pku.edu.cn)

---

## ⭐ Acknowledgements

This project is built on:

* SUMO
* OpenStreetMap
* Azure Maps / Amap APIs

```
