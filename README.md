# SatQuery AI: Drishti Spatial Intelligence (SIH 2026 - PS 26167)

**Problem Statement:** 26167 (ISRO/SAC)
**Title:** SatQuery AI: An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries
**Team Name:** Antariksh Astra

---

## 🌍 Overview

**Drishti Spatial AI** is a fully interactive, multimodal Vision-Language Assistant built for the **Indian Space Research Organisation (ISRO)** and **Space Applications Centre (SAC)**. It replaces complex GIS software with an intuitive, cinematic 3D globe interface powered by an agentic multimodal reasoning pipeline.

Our solution democratizes satellite intelligence, allowing non-experts, disaster response teams, and policymakers to query multi-spectral and SAR satellite constellations entirely through natural text and voice commands.

![Drishti Spatial AI Interface](https://raw.githubusercontent.com/anushreya0606/SAT_QUERY_AI/main/jarvis-god-eye/assets/preview.png) *(Preview of the cinematic God-Eye UI)*

---

## ✨ Key Features

### 1. Cinematic 3D "God-Eye" Globe Frontend
- **Interactive UI:** Built with Three.js, offering a military-grade, futuristic targeting interface.
- **Voice & Text Sandbox:** A persistent "Drishti Spatial AI" chat panel that supports text-to-speech (TTS) and speech-to-text.
- **Auto-Targeting:** The globe automatically flies to and zooms in on targets recognized by the AI or Nominatim Geocoding API.
- **Bi-Temporal Image Popups:** Displays "Before/After" satellite imagery overlays dynamically when analyzing changes.

### 2. Intelligent Agentic Pipeline
- **Conversational Intelligence:** Answers natural language queries, greets users based on system time, and handles conversational fallbacks.
- **Automated Workflow:** Translates complex queries (e.g., "Show me deforestation in Dehradun") into spatial coordinates, triggers the pipeline, and speaks the results.

### 3. FastAPI Agentic Backend
- **Python-powered Router:** A FastAPI server (astapi_server.py) acts as the brain, simulating VLM inference and returning geocodes, confidence scores, and strict execution traces.
- **SAR & Optical Fusion:** Built to cross-reference Cartosat optical with RISAT SAR backscatter.
- **Change Detection:** Built-in multimodal reasoning pipelines to detect urban encroachment, deforestation, and natural disasters.

### 4. ISRO Execution Trace Audit
- **Strict JSON Tracing:** A live "EXECUTION TRACE" panel renders real-time JSON logs showing step_id, module, and ction.
- **Transparency:** Demonstrates exactly how the VLM and schema validators are reasoning through spatial data, fulfilling ISRO's requirement for reproducible auditing.

---

## 🛠️ Architecture

### Frontend (/jarvis-god-eye)
- **HTML5/CSS3:** Custom HUD with JetBrains Mono and Orbitron fonts.
- **Vanilla JS + Three.js:** 3D globe rendering, atmospheric scattering, and smooth camera flying.
- **Web Speech API:** Native Text-to-Speech (TTS) and Speech Recognition.
- **Nominatim API:** Live geocoding for unrecognized generic queries (e.g., "India", "USA").

### Backend (astapi_server.py)
- **FastAPI:** High-performance async Python backend running on port 8000.
- **Agentic Fallback:** Dynamically processes textual intent and routes instructions to the frontend's spatial modules.

---

## 🚀 Quickstart Guide (Local Presentation)

To run the solution locally for the presentation:

### 1. Start the Agentic Backend
Open a terminal and start the Python FastAPI server:
`ash
# Ensure FastAPI and Uvicorn are installed
pip install fastapi uvicorn pydantic

# Run the server
python fastapi_server.py
`
*The server will start on http://localhost:8000.*

### 2. Start the 3D Frontend
Open a second terminal, navigate to the frontend directory, and start a local HTTP server:
`ash
cd jarvis-god-eye
python -m http.server 3000
`

### 3. Access the Dashboard
- Open your browser and navigate to: http://localhost:3000
- Click **"INITIALIZE UPLINK"** to trigger the boot sequence.
- Wait for the **"Drishti Spatial AI"** voice greeting.
- Click the bottom panel to expand the chat and ask:
  - *"Show me deforestation"*
  - *"Analyze SAR fusion"*
  - *"India"*

---

## 🛡️ License
Built for the Smart India Hackathon 2026. Codebase is restricted to evaluating jury members.