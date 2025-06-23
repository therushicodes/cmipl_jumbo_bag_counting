# Sack Counter with YOLOv8, DeepSORT & Hikvision RTSP

This Python-based application performs **real-time object detection and tracking** from a Hikvision RTSP stream using **YOLOv8** and **DeepSORT**. It detects jumbo sacks, tracks them, logs when they cross a virtual zone, and automatically sends email reports with CSV logs on a schedule.

---

## Features

- RTSP stream support for Hikvision cameras (main and sub stream)
- YOLOv8-based object detection
- DeepSORT real-time tracking
- Logging of sack crossing events with timestamps
- Scheduled email reports with CSV attachment
- Auto-reset of tracked IDs at configured times

---

## Setup Instructions

### 1. Clone the Repository

`git clone https://github.com/therushicodes/cmipl_jumbo_bag_counting.git`

`cd cmipl_jumbo_bag_counting`

### 2. Install dependencies

`pip install ultralytics opencv-python pandas schedule deep_sort_realtime`

### 3. Configure YOLOv8 Model
Update the model path in the code:
`model = YOLO('/path/to/your/best.pt')`

### 4. Camera Configuration
Edit these values in the script to match your Hikvision camera:
`username = "admin"`
`password = "your_camera_password"`
`ip_address = "192.168.x.x"`
`port = 554  # Usually 554 for RTSP`

Available RTSP URLs:

Main stream: rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/101

Sub stream: rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/102?

Alternate main: rtsp://<user>:<pass>@<ip>:554/h264/ch1/main/av_stream


### 5. Email Configuration

`sender_email = "youremail@gmail.com"`
`password = "your_app_password"`
`receivers = ["receiver1@domain.com", "receiver2@domain.com"]`

## How it works
YOLOv8 detects sacks.

1. DeepSORT tracks them across frames.
2. When a sack crosses the defined virtual zone, it is logged.
3. CSV saved at ~/Desktop/CSV_files/crossing_log.csv.
4. Emails are scheduled to send updates + CSV file.

## Authors
Developed at Sapien Robotics
Created by Rushikesh

