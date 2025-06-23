import cv2
import os
import pathlib
import numpy as np
import pandas as pd
import time
import schedule
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Camera credentials and IP
username = "admin"
password = "asif@1234"
ip_address = "192.168.3.123"
port = 554  # Default RTSP port for Hikvision cameras

# Load YOLO model and DeepSORT tracker
model = YOLO('/home/sapien1/cmipl_demo/runs/detect/train/weights/best.pt')
tracker = DeepSort(max_age=30)

# Setup the folder for csv file
output_folder = str(pathlib.Path.home() / "Desktop/CSV_files")
os.makedirs(output_folder, exist_ok=True)

# CSV for logging sack crossings
csv_output_path = os.path.join(output_folder, 'crossing_log.csv')
crossing_logs = []  # To store crossing event data

# Counting parameters
counted_ids = set()
unique_ids = set()
prev_positions = {}

# RTSP URL formats for Hikvision cameras
# Main stream (high quality)
rtsp_url_main = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/101"

# Sub stream (lower quality, good for preview)
rtsp_url_sub = f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/102?tcp"

def stream_camera(rtsp_url, window_name="Hikvision Camera"):
    """
    Stream video from Hikvision camera using OpenCV
    """
    # Create VideoCapture object with additional options for better compatibility
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set additional properties for better performance
    cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce load
    
    # For main stream, try to force lower resolution to avoid decoding issues
    if "101" in rtsp_url:  # Main stream
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 192)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 108)
    
    if not cap.isOpened():
        print(f"Error: Could not connect to camera at {rtsp_url}")
        print("Please check:")
        print("1. Camera IP address and credentials")
        print("2. Network connectivity")
        print("3. Camera RTSP settings")
        return False
    
    print(f"Successfully connected to camera: {rtsp_url}")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define rectangular zone for sack counting
    zone_width = 100
    zone_x1 = (frame_width // 2) - (zone_width)
    zone_x2 = (frame_width // 2) + (zone_width)
    zone_y1 = 0
    zone_y2 = frame_height

    
    frame_idx = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            frame_count += 1
            detections = []

            # Run YOLO detection every 1 second
            if frame_idx % fps == 0:
                results = model(frame, conf=0.3)[0]
                
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

                # Update DeepSORT tracker
                tracks = tracker.update_tracks(detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                unique_ids.add(track_id)

                # Draw bounding box and center
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Check for crossing zone
                if track_id in prev_positions:
                    prev_x = prev_positions[track_id]
                    
                    if ((zone_x1 <= prev_x <= zone_x2) and cx < zone_x1) and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        crossing_logs.append({
                            'Track_ID': track_id,
                            'Timestamp (s)': timestamp
                        })
                prev_positions[track_id] = cx

            # Draw rectangular zone
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
            # cv2.putText(frame, "Count Zone", (zone_x1 + 10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw count
            cv2.putText(frame, f"Unique sacks: {len(unique_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Sacks crossed: {len(counted_ids)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            frame_idx += 1
            # Display frame info on the image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"IP: {ip_address}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"hikvision_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

    
    except KeyboardInterrupt:
        print("\nStopping camera stream...")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        # Save all crossing logs to a single CSV
        if crossing_logs:
            df = pd.DataFrame(crossing_logs)
            df.to_csv(csv_output_path, index=False)
        print("Camera stream stopped")
    
    return True

# Function to reset both sets
def reset_tracking_ids():
    global unique_ids, counted_ids
    unique_ids.clear()
    counted_ids.clear()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tracking sets reset.")

def test_connection():
    """
    Test connection to both main and sub streams
    """
    print("Testing camera connection...")
    
    # Test main stream
    print("\n1. Testing main stream (high quality):")
    cap_main = cv2.VideoCapture(rtsp_url_main)
    if cap_main.isOpened():
        ret, frame = cap_main.read()
        if ret:
            print("âœ“ Main stream working")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("âœ— Main stream not receiving frames")
    else:
        print("âœ— Main stream connection failed")
    cap_main.release()
    
    # Test sub stream
    print("\n2. Testing sub stream (lower quality):")
    cap_sub = cv2.VideoCapture(rtsp_url_sub)
    if cap_sub.isOpened():
        ret, frame = cap_sub.read()
        if ret:
            print("âœ“ Sub stream working")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("âœ— Sub stream not receiving frames")
    else:
        print("âœ— Sub stream connection failed")
    cap_sub.release()

def send_email():
    subject = "Scheduled Email - Sack Counting"
    body = (
        f"This is an automatically scheduled email to inform about jumbo sacks.\n"
        f"The number of jumbo sacks crossed so far is {len(counted_ids)}.\n\n"
        "Please find the attached CSV log file."
    )

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receivers)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach CSV
    if os.path.exists(csv_output_path):
        with open(csv_output_path, "rb") as file:
            from email.mime.application import MIMEApplication
            attachment = MIMEApplication(file.read(), _subtype="csv")
            attachment.add_header('Content-Disposition', 'attachment', filename="crossing_log.csv")
            msg.attach(attachment)
    else:
        print("Warning: CSV file not found to attach.")

    # Send Email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receivers, msg.as_string())
        server.quit()
        print(f"Email sent at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print("Failed to send email:", e)

# List of times to send email (24-hour format)
times_to_send = ["10:00", "14:00", "18:00", "22:00"]

# Schedule emails at specified times
for t in times_to_send:
    schedule.every().day.at(t).do(send_email)
    schedule.every().day.at(t).do(reset_tracking_ids)
    print(f"Email scheduled at {t}")

if __name__ == "__main__":
    # Test connection
    test_connection()

    # Camera selection
    choice = input("Enter stream choice (1=Main, 2=Sub, 3=Alt Main) [default=2]: ").strip()
    if choice == "1":
        selected_url = rtsp_url_main
        window_name = "Hikvision Camera - Main Stream"
    elif choice == "3":
        selected_url = f"rtsp://{username}:{password}@{ip_address}:{port}/h264/ch1/main/av_stream"
        window_name = "Hikvision Camera - Alt Main Stream"
    else:
        selected_url = rtsp_url_sub
        window_name = "Hikvision Camera - Sub Stream"

    # List of times to send email (24-hour format)
    times_to_send = ["10:00", "14:00", "18:00", "22:00"]

    # Schedule emails and resets
    for t in times_to_send:
        schedule.every().day.at(t).do(send_email)
        schedule.every().day.at(t).do(reset_tracking_ids)
        print(f"Scheduled: email & reset at {t}")

    # Start camera stream
    stream_camera(selected_url, window_name)

# Email credentials and settings
sender_email = "sapientailscale@gmail.com"
receivers = ["aditya.patil@sapienrobotics.ai",
             "aryaman.bansal@sapienrobotics.ai",
             "sapien-collab-1@sapienrobotics.ai"]
password = "your_app_password" 


# Run the scheduler loop
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
