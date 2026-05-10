import random
import base64
import io
import math

PLASTIC_TYPES = [
    "Plastic Bottle",
    "Plastic Bag",
    "Plastic Fragment",
    "Fishing Net",
    "Foam/Styrofoam",
    "Plastic Cap",
    "Straw",
    "Plastic Film",
]

def simulate_detection(image_bytes: bytes, filename: str) -> dict:
    """
    Simulates YOLOv8-style plastic detection on a marine image.
    In a real deployment, replace this with actual YOLOv8 inference.
    
    Returns detection results with bounding boxes, confidence scores,
    and severity classification — matching real YOLO output format.
    """
    seed = sum(image_bytes[:100]) if len(image_bytes) >= 100 else len(image_bytes)
    random.seed(seed)

    plastic_count = random.choices(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        weights=[5, 8, 12, 15, 15, 12, 10, 8, 7, 5, 3]
    )[0]

    detections = []
    confidences = []

    for i in range(plastic_count):
        x1 = random.uniform(0.05, 0.7)
        y1 = random.uniform(0.05, 0.7)
        w = random.uniform(0.05, 0.25)
        h = random.uniform(0.04, 0.2)
        x2 = min(x1 + w, 0.95)
        y2 = min(y1 + h, 0.95)

        confidence = random.uniform(0.55, 0.97)
        confidences.append(confidence)

        label = random.choice(PLASTIC_TYPES)

        detections.append({
            "id": i + 1,
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": {
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3)
            }
        })

    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    if plastic_count == 0:
        severity = "None"
    elif plastic_count <= 2:
        severity = "Low"
    elif plastic_count <= 5:
        severity = "Medium"
    else:
        severity = "High"

    baseline_items = plastic_count + random.randint(3, 8)
    reduction_pct = round(((baseline_items - plastic_count) / baseline_items) * 100, 1) if baseline_items > 0 else 0

    return {
        "success": True,
        "filename": filename,
        "plastic_count": plastic_count,
        "avg_confidence": avg_confidence,
        "severity": severity,
        "detections": detections,
        "model": "YOLOv8n-marine-simulation",
        "baseline_estimate": baseline_items,
        "reduction_pct": reduction_pct,
        "note": "Simulated detection — replace detector.py with real YOLOv8 inference for production"
    }
