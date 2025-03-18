# Advanced Driver Assistance System (ADAS) Project

This project implements an ADAS system that fuses data from multiple sensors (camera and radar/LIDAR) to detect obstacles and make decisions to avoid collisions.

## Project Overview

The project is structured to address the following key components:

1. **ADAS Architecture**: System-level flowchart showing the integration of sensors and ML models
2. **Data Preprocessing and Correlation**: Processing camera and radar/LIDAR data
3. **Object Detection and Placement**: Correlating objects between sensors and identifying obstructions
4. **Decision Making**: Implementing collision avoidance mechanisms
5. **Efficiency Improvements**: Suggestions for improving prediction efficiency

## Project Structure

```
├── data/                      # Directory for datasets
├── models/                    # Trained models
├── src/                       # Source code
│   ├── architecture/          # ADAS architecture diagrams
│   ├── preprocessing/         # Data preprocessing modules
│   ├── object_detection/      # Object detection and correlation
│   ├── decision_making/       # Collision avoidance algorithms
│   └── visualization/         # Visualization utilities
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the required datasets (links provided in the documentation)
4. Run the preprocessing scripts
5. Execute the object detection and fusion modules
6. Visualize the results

## Datasets

This project uses open-source datasets that include camera and radar/LIDAR data. Links to these datasets will be provided in the documentation.

## Approach

### Data Preprocessing

- Camera data: Transfer learning with custom models to identify objects
- Radar/LIDAR data: Processing to determine field of view and object distances

### Sensor Fusion

- Correlation of objects between camera and radar/LIDAR data
- Integration of sensor data for accurate object placement

### Collision Avoidance

- Algorithms for detecting potential collisions
- Decision-making logic for triggering warnings and corrective actions

## Results

The project will include visualizations of:

- Object detection results
- Sensor fusion outcomes
- Collision prediction and avoidance scenarios

## Improvements

The documentation will include at least three suggestions for improving the efficiency of the prediction system.
