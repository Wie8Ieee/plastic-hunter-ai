from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import random
import time
import base64
import io
from datetime import datetime, timedelta
from database import init_db, save_detection, get_all_detections, get_stats
from detector import simulate_detection

app = FastAPI(title="Plastic Hunter AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/detect")
async def detect_plastic(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    
    result = simulate_detection(image_bytes, file.filename)
    
    lat = random.uniform(31.5, 36.9)
    lon = random.uniform(34.2, 36.6)
    
    save_detection(
        image_name=file.filename,
        plastic_count=result["plastic_count"],
        avg_confidence=result["avg_confidence"],
        latitude=lat,
        longitude=lon,
        severity=result["severity"]
    )
    
    result["latitude"] = lat
    result["longitude"] = lon
    result["timestamp"] = datetime.now().isoformat()
    
    return JSONResponse(result)

@app.get("/results")
async def get_results():
    results = get_all_detections()
    return JSONResponse({"detections": results})

@app.get("/stats")
async def get_statistics():
    stats = get_stats()
    return JSONResponse(stats)

@app.post("/demo")
async def run_demo():
    demo_images = [
        {"name": "beach_sample_01.jpg", "count": 7, "conf": 0.84, "severity": "High"},
        {"name": "ocean_surface_02.jpg", "count": 3, "conf": 0.71, "severity": "Medium"},
        {"name": "coastal_area_03.jpg", "count": 1, "conf": 0.62, "severity": "Low"},
        {"name": "underwater_04.jpg", "count": 5, "conf": 0.79, "severity": "Medium"},
        {"name": "harbor_05.jpg", "count": 9, "conf": 0.91, "severity": "High"},
    ]
    
    coastal_coords = [
        (32.08, 34.78), (31.76, 34.63), (32.93, 35.07),
        (33.27, 35.17), (31.52, 34.46)
    ]
    
    for i, img in enumerate(demo_images):
        lat, lon = coastal_coords[i]
        save_detection(
            image_name=img["name"],
            plastic_count=img["count"],
            avg_confidence=img["conf"],
            latitude=lat + random.uniform(-0.1, 0.1),
            longitude=lon + random.uniform(-0.1, 0.1),
            severity=img["severity"]
        )
        time.sleep(0.05)
    
    return JSONResponse({"message": "Demo data loaded successfully", "count": len(demo_images)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
