# EMOGO Backend - Emotion Vlog API Server

This project implements a production-ready **FastAPI Backend** with **MongoDB Atlas** for the EMOGO emotion tracking application. It provides RESTful APIs for mood data collection, video uploads, and data export functionality, deployed on **Render** platform.

[View Live Backend »](https://emogo-backend-shih-yunlin.onrender.com) · [View Data Dashboard »](https://emogo-backend-shih-yunlin.onrender.com/export)

<br />

## Table of Contents

1. [About The Project](#-about-the-project)
2. [Core Features](#-core-features)
3. [API Reference](#-api-reference)
4. [Database Schema](#-database-schema)
5. [Project Structure](#-project-structure)
6. [Getting Started](#-getting-started)
7. [Tech Stack](#-tech-stack)
8. [Deployment](#-deployment)

---

## 📖 About The Project

**EMOGO Backend** serves as the cloud-based data storage and management layer for the EMOGO mobile app, handling:

- Receipt of mood scores, GPS coordinates, and video uploads from the mobile frontend
- Persistent storage in **MongoDB Atlas** across three collections
- Beautiful data visualization dashboard for reviewing collected data
- Export capabilities in CSV and ZIP formats

### Why This Architecture?

*   **Scalable**: MongoDB Atlas provides cloud-hosted, scalable database infrastructure
*   **Async-First**: Built with FastAPI for high-performance async request handling
*   **Production-Ready**: Deployed on Render with proper CORS, error handling, and file management

---

## ✨ Core Features

| Module | Description |
| :--- | :--- |
| **📹 Video Upload** | Receives and stores **MP4 mood vlogs** with auto-generated filenames |
| **😊 Mood Collection** | Captures **1-5 scale mood scores** with timestamps |
| **📍 GPS Storage** | Stores **latitude/longitude** coordinates with accuracy metrics |
| **📊 Data Dashboard** | Interactive **HTML dashboard** for viewing all collected data |
| **📥 Data Export** | Download data as **CSV files** or **ZIP archives** |
| **🔌 CORS Enabled** | Full cross-origin support for mobile app integration |

---

## 📸 Screenshots

<p align="center">
  <img src="./assets/app_home.png" alt="EMOGO App Home" width="250">
</p>
<p align="center"><em>Mobile App - Home Screen (BlueStacks)</em></p>

<p align="center">
  <img src="./assets/dashboard.png" alt="EMOGO Dashboard" width="600">
</p>
<p align="center"><em>Backend Dashboard - Real-time Data Overview</em></p>

<p align="center">
  <img src="./assets/gps_data.png" alt="GPS Location Data" width="600">
</p>
<p align="center"><em>GPS Data Export Page</em></p>

---

## 📡 API Reference

### Core Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | GET | Health check with animated landing page |
| `/api/moods` | POST | Upload mood record (FormData: `mood_score`, `video`, `latitude`, `longitude`, `location_accuracy`, `timestamp`) |

### Data Export Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/export` | GET | Dashboard with data overview |
| `/export/vlog` | GET | View all vlogs with video preview |
| `/export/sentiments` | GET | View mood score records |
| `/export/gps` | GET | View GPS location records |

### File Download Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/download/{filename}` | GET | Download specific video file |
| `/download-all` | GET | Download all videos as ZIP |
| `/download/sentiments-csv` | GET | Download mood data as CSV |
| `/download/gps-csv` | GET | Download GPS data as CSV |

### Static Files

| Path | Description |
| :--- | :--- |
| `/uploads/{filename}` | Serve uploaded video files |

---

## 🗄️ Database Schema

The application uses MongoDB with three collections:

### 1. `vlog` Collection
```json
{
  "_id": "ObjectId",
  "mood_score": 1-5,
  "video_url": "/uploads/filename.mp4",
  "timestamp": "ISO datetime",
  "created_at": "ISO datetime"
}
```

### 2. `sentiments` Collection
```json
{
  "_id": "ObjectId",
  "mood_score": 1-5,
  "timestamp": "ISO datetime",
  "created_at": "ISO datetime"
}
```

### 3. `gps` Collection
```json
{
  "_id": "ObjectId",
  "latitude": float,
  "longitude": float,
  "accuracy": float,
  "timestamp": "ISO datetime",
  "created_at": "ISO datetime"
}
```

---

## 📂 Project Structure

```plaintext
08-web-emogo-backend/
├── 🐍 main.py               # FastAPI Application & All Endpoints
├── 📋 requirements.txt      # Python Dependencies
├── ⚙️ render.yaml           # Render Deployment Configuration
├── 🔐 .env                  # Environment Variables (not in git)
├── 📦 *.aab                 # Android App Bundle (see note below)
├── 📁 uploads/              # Video File Storage Directory
│   └── *.mp4               # Uploaded Mood Vlogs
└── 📱 frontend/             # [Optional] Expo Mobile App
    ├── app/                # React Native Screens
    ├── components/         # Reusable UI Components
    ├── database/           # SQLite Local Storage
    └── utils/              # Helper Functions
```

### 📦 About the `.aab` File

The `.aab` (Android App Bundle) file is the **Google Play Store upload format**, generated by Expo EAS Build.

| Format | Purpose | Can Install Directly? |
| :--- | :--- | :---: |
| `.apk` | Direct installation on device/emulator | ✅ Yes |
| `.aab` | Upload to Google Play Store for distribution | ❌ No |

> [!NOTE]
> To test locally on BlueStacks or Android devices, use the **APK** build below. The AAB is designed for Play Store submission only.
>
> 📲 **Download Android APK (Test Build)**: [Install from Expo](https://expo.dev/accounts/shihyun_lin/projects/expo-router-mwe/builds/c935be2b-6afc-4f77-90ac-f4885e38f3c0)

---

## 🚀 Getting Started

### Prerequisites

*   **Python**: 3.8+
*   **MongoDB**: Local instance or MongoDB Atlas account

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd 08-web-emogo-backend
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    ```bash
    # Create .env file with:
    DATABASE_URL=mongodb+srv://username:password@cluster.mongodb.net/?appName=YourApp
    ```

4.  **Start Development Server**
    ```bash
    uvicorn main:app --reload
    ```

Server will start at `http://localhost:8000`

### Testing the API

```bash
# Health check
curl http://localhost:8000/

# View dashboard
open http://localhost:8000/export
```

---

## 🛠️ Tech Stack

*   **Framework**: FastAPI 0.104.1
*   **Database**: MongoDB Atlas (Cloud)
*   **Async Driver**: Motor (MongoDB Async Driver)
*   **Deployment**: Render (PaaS)
*   **File Storage**: Local filesystem (`uploads/` directory)
*   **Environment**: python-dotenv

---

## 🚀 Deployment

### Render Configuration

| Setting | Value |
| :--- | :--- |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Environment Variable** | `DATABASE_URL` (set in Render Dashboard) |

### Known Limitations

> [!WARNING]
> **Ephemeral Storage**: Uploaded videos are stored in the local filesystem. On Render's free tier, files are cleared when the service restarts.

**Recommended Solution**: For production use, implement MongoDB GridFS or cloud storage (S3, Cloudinary).

---

## 📥 Export Pages (For Reviewers)

| Data Type | URL | Features |
| :--- | :--- | :--- |
| **Vlog Videos** | [/export/vlog](https://emogo-backend-shih-yunlin.onrender.com/export/vlog) | Video preview, download, batch ZIP |
| **Mood Data** | [/export/sentiments](https://emogo-backend-shih-yunlin.onrender.com/export/sentiments) | Score history, CSV export |
| **GPS Data** | [/export/gps](https://emogo-backend-shih-yunlin.onrender.com/export/gps) | Location records, CSV export |

---

## ✅ Assignment Compliance

- ✅ **[Required]** Backend deployed on public server (Render) using FastAPI + MongoDB
- ✅ **[Required]** Data export pages for all three data types (vlogs, sentiments, GPS)
- ✅ **[Optional]** Frontend modified to upload data to deployed backend

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

