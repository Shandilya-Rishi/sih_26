# 🚍 URBAN-EYE Nexus

### **Every Bus. Every Road. One Intelligent City.**

> **SIH26-124** — AI-powered mobile urban sensing platform for public transport fleets.

URBAN-EYE Nexus transforms ordinary public transport buses into **mobile Edge-AI urban sensing units**. Instead of using bus-mounted cameras only for passive recording, the platform turns their video streams into structured, geotagged intelligence about road conditions, traffic, infrastructure and public safety.

---

## 🎯 What We Are Building

```text
┌──────────────────────────────────────────────────────────────┐
│                         PUBLIC BUS                           │
│                                                              │
│  Cameras + GPS + IMU                                         │
│          │                                                   │
│          ▼                                                   │
│      🧠 EDGE AI                                              │
│  Detection • Tracking • Fusion • Filtering                  │
│          │                                                   │
│          ▼                                                   │
│    Meaningful Urban Events                                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
                 4G / 5G / Wi-Fi
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  CENTRAL INTELLIGENCE PLATFORM               │
│                                                              │
│ GIS Map • Traffic Analytics • Road Health • Reports          │
│ Prediction • Event Correlation • Maintenance Prioritization  │
└──────────────────────────────────────────────────────────────┘
```

### Core idea

The system is **not just a collection of computer-vision models**.

It is designed as a distributed urban intelligence network where:

- 🚌 buses act as mobile sensing nodes
- 👁️ computer vision converts video into observations
- 📍 GPS attaches observations to real-world locations
- 🔄 repeated observations can verify persistent problems
- 🧠 an intelligence layer converts detections into events, severity and risk
- 📡 Edge AI filters unnecessary data before transmission
- 🗺️ the central platform aggregates fleet-wide intelligence

---

# 🚀 Current MVP Status

We are developing the AI/Edge subsystem incrementally.

| Milestone | Status |
|---|:---:|
| Project foundation | ✅ |
| M1 — YOLO object detection | ✅ |
| M2 — Vehicle tracking + unique counting | ✅ |
| M2.5 — Traffic density intelligence | ✅ |
| M3 — Road damage / pothole model | 🔜 |
| M4 — Number plate detection + OCR | 🔜 |
| M5 — Pedestrian safety intelligence | 🔜 |
| M6 — GPS/event pipeline | 🔜 |
| M7 — Edge filtering & event aggregation | 🔜 |
| M8 — Multi-camera fusion | 🔜 |
| M9 — TensorRT / DeepStream optimization | 🔜 |
| M10 — Jetson edge deployment | 🔜 |
| Central web-platform integration | 🔜 |

> **Note:** The central website/dashboard is being developed separately by the web-platform team. This repository focuses primarily on the AI/ML and Edge-AI subsystem.

---

# 🧠 Planned AI Capabilities

### 🛣️ Road & Infrastructure Intelligence
- Potholes and road-surface damage
- Damaged roads
- Missing/damaged dividers
- Zebra crossing detection
- Traffic-sign detection
- Infrastructure deficiencies
- Waterlogging
- Longitudinal monitoring of road-condition changes

### 🚗 Traffic Intelligence
- Vehicle detection
- Vehicle classification
- Vehicle tracking
- Unique vehicle counting
- Traffic density estimation
- Vehicle flow and direction
- Bottleneck identification
- Route-delay analysis
- Future congestion prediction

### 🚸 Public Safety
- Pedestrian detection
- Vulnerable-road-user analysis
- School-child crossing risk
- Vehicle/pedestrian trajectory analysis
- Time-to-collision based risk estimation

### 🚨 Incident Intelligence
- Rash-driving indicators
- Hit-and-run event detection
- Vehicle tracking
- Registration-number extraction
- Timestamp + GPS evidence
- Cross-bus observation correlation for authorized investigations

---

# ⭐ What Makes URBAN-EYE Different

The project deliberately goes beyond:

> **“Run YOLO on bus video.”**

The intended differentiators are:

### 🚌 1. Bus-as-a-Sensor Network
Every bus becomes a mobile observation node, extending sensing coverage across the road network without requiring fixed cameras everywhere.

### 🔄 2. Multi-Bus Verification
A road issue detected by one bus can later be independently observed by other buses at approximately the same location.

```text
Bus A → pothole detected
Bus B → same location
Bus C → same location
              ↓
       VERIFIED EVENT
```

### 📈 3. Dynamic Road Health
Road segments can receive continuously updated condition scores based on observed defects, traffic exposure, infrastructure state and safety risk.

### 🧠 4. Predictive + Prescriptive Intelligence
The system is intended to answer not only:

> **What happened?**

but also:

> **What is likely to happen next?**

and:

> **What should authorities prioritize?**

### 📡 5. Edge-First Processing
Raw video does not need to be continuously uploaded. The edge node can process locally and transmit meaningful events and evidence.

---

# 🔬 Current AI Pipeline

The current prototype has already established the first stages of the perception pipeline:

```text
Video
  ↓
OpenCV ingestion
  ↓
YOLO object detection
  ↓
ByteTrack tracking
  ↓
Vehicle IDs
  ↓
Traffic-density estimation
  ↓
Structured intelligence
```

The architecture will progressively evolve toward:

```text
Camera Streams
      ↓
Edge Preprocessing
      ↓
Multi-Model Perception
      ├── Vehicle Detection
      ├── Road Damage
      ├── Pedestrian Analysis
      ├── Traffic Signs
      └── ANPR / OCR
      ↓
Tracking + Sensor Fusion
      ↓
Event Intelligence
      ↓
Confidence + Severity + Risk
      ↓
Temporal Deduplication
      ↓
Edge Filtering
      ↓
Structured Event
      ↓
Central Platform
```

---

# 🧩 Repository Structure

```text
sih_26/
│
├── src/
│   ├── detection/          # Object and road-feature detection
│   ├── tracking/           # Vehicle/object tracking
│   ├── intelligence/       # Traffic, risk and event intelligence
│   ├── geolocation/        # GPS/geospatial processing
│   ├── ingestion/          # Video/camera input
│   ├── preprocessing/      # Frame/data preprocessing
│   ├── edge/               # Edge filtering and event handling
│   └── utils/              # Shared utilities
│
├── notebooks/              # ML experiments and training notebooks
├── datasets/               # Local datasets (not committed)
├── models/                 # Model weights (not committed)
├── configs/                # Configuration
├── outputs/                # Generated results/videos
├── tests/                  # Automated tests
├── scripts/                # Runnable project scripts
├── docs/                   # Technical documentation
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Technology Direction

### Current development

- **Python**
- **OpenCV**
- **Ultralytics YOLO**
- **ByteTrack**
- **Pytest**

### Planned / evaluation stage

- PyTorch
- PaddleOCR / OCR pipeline
- GPS + IMU fusion
- ONNX
- TensorRT
- NVIDIA DeepStream
- NVIDIA Jetson Orin Nano Super
- PostGIS
- FastAPI
- MQTT / WebSockets

The final stack will be chosen according to measured prototype performance, deployment constraints and license compatibility.

---

# ⚙️ Development Philosophy

This repository is being built as an **incremental, testable AI system**.

### Principles

**1. Build one working vertical slice at a time**

Every milestone should produce something executable.

**2. Separate perception from intelligence**

A detector should answer:

> “What did I see?”

The intelligence layer should answer:

> “What does it mean?”

**3. Separate training from deployment**

Training experiments belong in notebooks/Colab; the deployable inference pipeline belongs in `src/`.

**4. Design for Edge AI from the beginning**

The final system must be able to operate under constrained bandwidth and intermittent connectivity.

**5. Test before extending**

Every milestone gets a test and a Git checkpoint before the next major change.

---

# 🧪 Development Workflow

```text
1. Research / dataset selection
          ↓
2. Experiment in Google Colab
          ↓
3. Train / evaluate model
          ↓
4. Export model artifact
          ↓
5. Integrate into src/
          ↓
6. Test locally
          ↓
7. Build Edge-AI logic
          ↓
8. Optimize inference
          ↓
9. Deploy to Jetson
          ↓
10. Connect to central platform
```

---

# 📦 Data Policy

Datasets and trained model weights can become very large.

Therefore:

- raw datasets are kept outside Git
- generated videos stay outside Git
- model weight files stay outside Git
- only code, configuration, documentation and lightweight metadata should be committed

The `.gitignore` file is configured accordingly.

---

# 🎬 MVP Demonstration Goal

The AI subsystem should ultimately be able to demonstrate a pipeline such as:

```text
Bus Camera Video
       ↓
Vehicle / Road AI
       ↓
Tracking
       ↓
Road / Traffic Event
       ↓
Confidence + Severity
       ↓
GPS
       ↓
Edge Filtering
       ↓
JSON Event
       ↓
Central Web Platform
```

Example event:

```json
{
  "event_type": "ROAD_DEFECT",
  "subtype": "POTHOLE",
  "confidence": 0.94,
  "severity": "HIGH",
  "bus_id": "BUS_014",
  "timestamp": "2026-09-05T15:30:00",
  "gps": {
    "lat": 30.1234,
    "lon": 76.5678
  }
}
```

The website team can then consume this event to create:

- 🗺️ GIS markers
- 🚨 alerts
- 📊 dashboards
- 📋 incident reports
- 📈 road-health analytics

---

# 🏆 Long-Term Vision

URBAN-EYE Nexus aims to create a **city-scale mobile sensing network** using infrastructure cities already operate.

```text
              🚌
          🚌       🚌
      🚌               🚌
          🚌       🚌
              🚌
               │
               ▼
       DISTRIBUTED EDGE AI
               │
               ▼
      CENTRAL URBAN INTELLIGENCE
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
     Roads   Traffic   Safety
       │       │        │
       └───────┼────────┘
               ▼
       BETTER CITY ACTION
```

> **The goal is not to collect more video.  
> The goal is to turn the video cities already have into actionable intelligence.**

---

## 👥 Project Context

**Project:** URBAN-EYE Nexus  
**SIH Problem Statement:** SIH26-124  
**Repository:** `sih_26`  
**Focus of this repository:** AI / ML / Computer Vision / Edge-AI subsystem

---

## 📌 Current Focus

**M1:** YOLO object detection ✅  
**M2:** Vehicle tracking + unique counting ✅  
**M2.5:** Traffic density intelligence ✅  
**Next major milestone:** **M3 — Custom road-damage / pothole model**
