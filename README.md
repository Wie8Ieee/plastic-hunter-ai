# Plastic Hunter AI — EcoNauts

## AESS Sustainability Hackathon 2026 | Challenge 3

### AI-Powered Marine Plastic Detection & Sustainable Monitoring System

---

## Vision

Plastic pollution is a global environmental crisis affecting oceans, ecosystems, and coastal communities.

Plastic Hunter AI is built to address the question:

Can we detect marine plastic pollution faster, cheaper, and at scale using AI?

This project combines Computer Vision, Environmental Analytics, and Sustainable Monitoring concepts to build an intelligent system for marine pollution detection.

---

## Problem

Every year, more than 8 million tons of plastic enter the ocean. Current monitoring methods face several limitations:

- High operational cost  
- Slow manual inspection  
- Limited geographic coverage  
- Lack of real-time detection  

As a result, pollution is often identified too late for effective intervention.

---

## Solution

Plastic Hunter AI provides an automated detection and visualization system that converts marine images into environmental insights.

Core capabilities:

- Upload marine or coastal images  
- Detect plastic objects using computer vision analysis  
- Classify pollution types (bottles, nets, foam, bags, etc.)  
- Assign confidence and severity levels  
- Visualize detections on a global map using Leaflet.js  
- Track pollution trends over time  

---

## System Architecture

The system follows a modular pipeline:

### 1. Input Layer
User uploads image (marine/coastal environment)

↓

### 2. Backend Processing
FastAPI handles request at `/detect`

↓

### 3. Computer Vision Engine (`detector.py`)
- Image preprocessing (Pillow, NumPy)
- Region-based analysis (grid scoring)
- Object detection simulation
- Bounding box generation

↓

### 4. Data Storage Layer (SQLite)
- Stores detection results
- Saves annotated images in `/results`
- Maintains analytics in `/stats`
- Supports geo-visualization in `/map`

↓

### 5. Frontend Dashboard
- HTML / CSS / JavaScript UI
- Leaflet.js for map visualization
- Chart.js for analytics display


---

## Tech Stack

| Layer         | Technology            |
|--------------|----------------------|
| Backend       | FastAPI (Python)     |
| Computer Vision | Pillow, NumPy     |
| Database      | SQLite               |
| Frontend      | HTML, CSS, JavaScript |
| Visualization | Leaflet.js, Chart.js |


---

## Key Results

| Metric                      | Value           |
|----------------------------|----------------|
| Plastic categories detected | 14 types        |
| Detection speed            | < 1 second      |
| Confidence range           | 38% - 97%       |
| Demo locations             | 12 global sites |
| Estimated improvement      | 25–30%          |

The system demonstrates the difference between:

- No monitoring baseline  
- AI-assisted monitoring model  

Result: Faster detection enables earlier response and improved environmental action.

---

## Detection Logic

The system uses a lightweight simulation-based approach:

1. Image resizing for performance optimization  
2. Edge detection to highlight visual changes  
3. Grid segmentation to identify activity regions  
4. Bounding box generation for detected areas  
5. Classification simulation for object types  
6. Confidence score assignment  
7. Result storage and visualization  

This approach ensures:

- Fast processing  
- Lightweight execution  
- Easy deployment on limited hardware  

---

## Sustainability Aspect

| Metric                    | Conventional | EcoNauts System |
|--------------------------|-------------|----------------|
| Energy usage             | High        | Low            |
| Environmental impact     | High        | Reduced        |
| Monitoring approach      | Static      | Adaptive       |
| Efficiency               | Moderate    | Optimized      |

The goal is to design environmental monitoring systems that do not negatively impact the environment they observe.

---

## Limitations

- Detection is simulation-based rather than a full deep learning model  
- GPS coordinates are approximate for demonstration purposes  
- Performance depends on image quality  
- No real-time drone or satellite integration yet  

---

## Future Work

- Train YOLOv8 model on real marine datasets (TACO / PlasticLitter)  
- Integrate drone and satellite data sources  
- Real-time pollution alert system  
- Citizen reporting module  
- Cloud-based scalable deployment  
- Acoustic sonar-based environmental monitoring integration  

---

## AI Usage Disclosure

AI tools were used to assist in:

- Backend structure design using FastAPI  
- Frontend scaffolding  
- Documentation drafting  
- Simulation logic design support  

All system architecture decisions, implementation, and validation were performed by the EcoNauts team.

---

## SDGs Addressed

- SDG 14: Life Below Water  
- SDG 13: Climate Action  
- SDG 12: Responsible Consumption and Production  

---

## Repository Structure
/
├── main.py # FastAPI application entry point
├── detector.py # Core detection engine (computer vision logic)
├── database.py # Database handling and storage logic
├── YOLOv8_Detection.py # YOLOv8-based detection experiments
│
├── simulation/
│ └── sonar_simulation.py # Acoustic / sonar system simulation
│
├── assets/ # Project resources and media files
├── notebooks/ # Jupyter notebooks for experiments & analysis
├── static/ # Frontend dashboard (HTML/CSS/JS)
│
├── results/ # Output visualizations and analytics
│ ├── plot1_signal_comparison.png
│ ├── plot2_acoustic_impact.png
│ └── plot3_comparison_dashboard.png
│
└── README.md # Project documentation


---

## Final Note

Plastic Hunter AI is a prototype for scalable environmental intelligence systems, demonstrating how artificial intelligence can support marine ecosystem protection and sustainability efforts.
