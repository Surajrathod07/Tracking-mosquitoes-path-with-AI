# 🦟 Tracking Mosquitoes Path with AI

This project uses AI (YOLOv8 + DeepSORT) to **detect and track mosquitoes in real-time** from video input, helping analyze mosquito movement for research or pest control automation.

## 🚀 Features

- Real-time mosquito detection using YOLOv8
- Identity-based tracking with DeepSORT
- Works on video files or live camera input
- Outputs video with mosquito ID paths visualized

## 🧠 Tech Stack

- Python
- YOLOv8 (Ultralytics)
- DeepSORT (tracking)
- OpenCV
- Git LFS for large models

## 📁 Project Structure

```
Tracking Mosquitoes path with AI/
├── Script/
│   ├── detect.py
│   ├── track.py
│   └── ...
├── Videos/
│   └── input.mp4
├── Weights/
│   └── best.pt
├── mosquito_env/ (ignored)
├── requirements.txt
└── README.md
```

## 🛠 How to Run

1. Clone this repo
2. Set up Python environment  
   `pip install -r requirements.txt`
3. Run detection or tracking script  
   `python Script/track.py --source Videos/input.mp4`

## 📦 Model + Sample Video

> ⚠️ GitHub has a 100MB file limit. Large files like the trained model (`best.pt`) and videos are stored on Google Drive.

📥 Download model & video files:  
👉 [Google Drive Link](https://drive.google.com/...)

## 👨‍💻 Author

- Suraj Rathod  
- Guided by: Arvind Sir, Chandan Sir  
- Organization: CloudXcent Innovations

## 📄 License

Free to use for educational and research purposes.
