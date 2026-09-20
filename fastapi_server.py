# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio

app = FastAPI(title="ANTARIKSH ASTRA SPATIAL INTELLIGENCE SERVER")

# Enable CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow the Vercel frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import uuid
from datetime import datetime
from schema.trace import ExecutionTraceStep

# DOFA Mock setup (to replace the banned Cloud APIs)
# We simulate a DOFA (Dynamic Wavelength) model fallback instead of using Gemini
def generate_dofa_mock_trace(query: str, intent: str, confidence: float) -> dict:
    step = ExecutionTraceStep(
        step_id=f"DOFA_INFERENCE_{uuid.uuid4().hex[:6]}",
        module="DOFA_VLM_ENCODER",
        action="multimodal_reasoning",
        inputs={"query": query, "sensor": "sensor-agnostic"},
        outputs={"intent": intent, "confidence": confidence}
    )
    # Return as dict so FastAPI can serialize it easily
    return step.model_dump()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: QueryRequest):
    query = request.query.lower()
    
    # Simulate Agentic Routing Delay
    await asyncio.sleep(1)
    
    if "deforestation" in query or "change" in query:
        trace_data = generate_dofa_mock_trace(query, "change_detection", 0.94)
        return {
            "status": "success",
            "lat": 20.5937,
            "lng": 78.9629,
            "target_name": "DEFORESTATION FRONT, INDIA",
            "bbox": [20.5900, 78.9600, 20.5974, 78.9658],
            "trace": trace_data,
            "result": "RESULT: 2.4 SQ KM FOREST LOSS DETECTED. CONFIDENCE: 0.94. TRACE LOGGED."
        }
    elif "sar" in query or "cartosat" in query or "fusion" in query or "construction" in query:
        trace_data = generate_dofa_mock_trace(query, "sar_optical_fusion", 0.88)
        return {
            "status": "success",
            "lat": 28.6139,
            "lng": 77.2090,
            "target_name": "RISAT-CARTOSAT ALIGNMENT, NEW DELHI",
            "bbox": [28.6100, 77.2050, 28.6178, 77.2130],
            "trace": trace_data,
            "result": "RESULT: UNAUTHORIZED CONSTRUCTION IDENTIFIED. CONFIDENCE: 0.88. TRACE LOGGED."
        }
    else:
        # Dynamic Fallback using Local DOFA-VLM Mock (Replacing Gemini!)
        ai_response_text = f"RESULT: ANALYSIS COMPLETE FOR '{request.query.upper()}'. NO ANOMALIES."
        trace_data = generate_dofa_mock_trace(query, "general_query", 0.75)
        
        return {
            "status": "success",
            "action": "geocode",
            "trace": trace_data,
            "result": ai_response_text
        }

if __name__ == '__main__':
    import uvicorn
    print("Starting ANTARIKSH ASTRA backend on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
