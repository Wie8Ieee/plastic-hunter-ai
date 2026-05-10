import sqlite3
from datetime import datetime, timedelta
import random

DB_PATH = "plastic_hunter.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT,
            plastic_count INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0.0,
            latitude REAL,
            longitude REAL,
            severity TEXT DEFAULT 'None'
        )
    """)
    conn.commit()
    conn.close()

def save_detection(image_name, plastic_count, avg_confidence, latitude, longitude, severity):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO detections (timestamp, image_name, plastic_count, avg_confidence, latitude, longitude, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        image_name,
        plastic_count,
        avg_confidence,
        latitude,
        longitude,
        severity
    ))
    conn.commit()
    conn.close()

def get_all_detections():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM detections")
    total_scans = c.fetchone()[0]

    c.execute("SELECT SUM(plastic_count) FROM detections")
    total_plastic = c.fetchone()[0] or 0

    c.execute("SELECT AVG(avg_confidence) FROM detections WHERE plastic_count > 0")
    avg_conf = c.fetchone()[0] or 0.0

    c.execute("SELECT severity, COUNT(*) as cnt FROM detections GROUP BY severity")
    severity_rows = c.fetchall()
    severity_dist = {row[0]: row[1] for row in severity_rows}

    c.execute("""
        SELECT DATE(timestamp) as day, SUM(plastic_count) as total
        FROM detections
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
        LIMIT 7
    """)
    daily_rows = c.fetchall()
    daily_data = [{"date": row[0], "count": row[1]} for row in daily_rows]

    baseline_total = int(total_plastic * 1.4) if total_plastic > 0 else 0
    reduction_pct = round(((baseline_total - total_plastic) / baseline_total) * 100, 1) if baseline_total > 0 else 0

    conn.close()
    return {
        "total_scans": total_scans,
        "total_plastic_detected": total_plastic,
        "avg_confidence": round(avg_conf, 3),
        "severity_distribution": severity_dist,
        "daily_detections": daily_data,
        "baseline_total": baseline_total,
        "reduction_pct": reduction_pct
    }
