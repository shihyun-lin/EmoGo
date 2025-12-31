# Computer Vision: Taiwanese Emotion Recognition Analysis

This project implements a comprehensive **Computer Vision (CV) Analysis** pipeline to evaluate and improve emotion recognition performance on East Asian faces. It specifically targets the **Taiwanese Facial Expression Image Database (TFEID)** to address the "Cross-Race Effect" found in standard Western-centric models.

[View Analysis Notebook »](./03_HW14_taiwanese_faces/R13546008_HW14.ipynb) · [View PDF Report »](./03_HW14_taiwanese_faces/R13546008_HW14.pdf)

<br />

## Table of Contents

1. [About The Project](#-about-the-project)
2. [Key Findings](#-key-findings)
3. [Project Structure](#-project-structure)
4. [Methodology](#-methodology)
5. [Experimental Results](#-experimental-results)
6. [Getting Started](#-getting-started)
7. [Tech Stack](#-tech-stack)
8. [Privacy Notice](#-privacy-notice)
9. [License](#-license)

---

## 📖 About The Project

**Emotion Recognition Analysis** serves as a validation and optimization study for the EMOGO project's core AI features. It aims to answer a critical question: *Can we improve emotion detection accuracy for Taiwanese users using custom training?*

The project compares industry-standard "Zero-shot" models against a custom-trained classifier, identifying significant performance gains when using race-specific training data.

### Research Goals

*   **Benchmark**: Evaluate performance of global models (DeepFace, FER) on Taiwanese faces.
*   **Improve**: Develop a custom fusion model to surpass baseline accuracy.
*   **Validate**: Test the models on real-world video footage to assess practical viability.

---

## ✨ Key Findings

| Metric | Result |
| :--- | :--- |
| **📉 Baseline Accuracy** | **70.8%** (DeepFace) - Struggles with Asian facial features due to data bias. |
| **🚀 Custom Model** | **91.3%** - Achieved by fine-tuning on TFEID data. |
| **💡 Core Insight** | **Fine-tuning is essential** to mitigate the Cross-Race Effect for Asian users. |
| **🎥 Video Analysis** | "In-the-wild" video requires **temporal smoothing**; static accuracy does not map 1:1 to video stability. |

---

## 📂 Project Structure

This directory (`09-cv-recognition-examples`) contains several experiments, with the main analysis located in `03_HW14_taiwanese_faces`.

```plaintext
09-cv-recognition-examples/
├── 📁 01_info_13_examples/    # Basic CV examples and info
├── 📁 02_old_13_exercises/    # Previous course exercises
├── 📁 misc/                   # Miscellaneous scripts
└── 📁 03_HW14_taiwanese_faces/
    ├── 🐍 R13546008_HW14.ipynb  # Core Analysis Notebook (Preprocessing, Training, Evaluation)
    ├── 📄 R13546008_HW14.pdf    # PDF Report of the Analysis
    ├── 🚫 Taiwanese/            # [Ignored] Raw Image Dataset
    └── 🚫 *.mp4                 # [Ignored] Testing Videos (vlog.mp4)
```

> [!NOTE]
> **Data Privacy**: All large media files and sensitive datasets (especially TFEID images) are excluded via `.gitignore` to protect privacy.

---

## 🛠️ Methodology

### 1. Zero-shot Evaluation
We evaluated two pre-trained models without any fine-tuning:
*   **DeepFace (VGG-Face)**: Standard deep learning facial recognition.
*   **FER (mtcnn)**: Uses MTCNN for detection + Keras for classification.

### 2. Custom Model Training (Feature Fusion)
We trained a **Logistic Regression** classifier using a fused feature set (Total: 519 dimensions):

| Feature Source | Dimensions | Description |
| :--- | :--- | :--- |
| **DeepFace** | 512 | VGG-Face Embeddings representing facial features |
| **FER** | 7 | Probability scores for 7 emotions (Angry, Happy, Neutral, etc.) |

---

## 📊 Experimental Results

Tested on a filtered validation set of 106 "clean" images.

<div align="center">

| Model | Accuracy | Performance Note |
| :--- | :---: | :--- |
| **DeepFace** | `70.8%` | Global baseline. Good on Happy, poor on Sad/Angry. |
| **FER** | `80.6%` | Better baseline. Still struggles with minority classes. |
| **Custom LR** | `91.3%` | **Best Performer**. Significantly adapts to Taiwanese features. |

</div>

### Static vs. Dynamic (Video)
*   **Static Images**: High precision possible with domain adaptation.
*   **Video**: Requires **temporal smoothing**. Frame-by-frame prediction allows for "jitter" (rapid emotion switching), which degrades user experience.

---

## 🚀 Getting Started

### Prerequisites

*   **Python**: 3.8+
*   **Jupyter Notebook**: For running `.ipynb` files.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd 09-cv-recognition-examples/03_HW14_taiwanese_faces
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy opencv-python scikit-learn deepface fer
    ```

3.  **Run Analysis**
    Open `R13546008_HW14.ipynb` in Jupyter Notebook or VS Code.

---

## �️ Tech Stack

*   **Language**: Python 3.x
*   **Computer Vision**: OpenCV (`cv2`), MTCNN
*   **Machine Learning**: Scikit-learn (Logistic Regression)
*   **Deep Learning**: DeepFace (VGG-Face), FER (Keras)
*   **Data Processing**: Pandas, NumPy
*   **Data**: Taiwanese Facial Expression Image Database (TFEID)

---

## ⚖️ Privacy Notice

> [!IMPORTANT]
> **Data Privacy Compliance**
>
> This repository strictly adheres to data privacy standards.
> *   ✅ **NO** original images from the TFEID dataset are included.
> *   ✅ **NO** personal video files (`vlog.mp4`, etc.) are included.
> *   The repository contains **only code and analysis reports**.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
