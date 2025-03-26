import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.utils.plotting import Annotator

# Define class lists and behavior mapping with point deductions
class_names_employee = ['At-Desk-NotWorking', 'At-Desk-Working', 'Sleeping', 'Standing-NotWorking', 'Standing-Working', 'Walking']
class_names_general = ["person"]
point_deductions = {0: 5, 2: 10, 3: 5}  # Points to deduct for each undesired behavior

# Initialize models and tracker
model_employee = YOLO('best.pt')  # Employee detection and behavior model
model_general = YOLO('yolov8n.pt')  # General detection model (YOLOv8n)
person_tracker = DeepSort(max_age=50, n_init=50, max_iou_distance=0.5)
behavior_tracker = DeepSort(max_age=50, n_init=50, max_iou_distance=0.5)

# Color mapping for models
model_colors = {
    model_employee: (255, 0, 0),  # Red for employee model
    model_general: (0, 255, 0),   # Green for general model (YOLOv8n)
}

# Initialize tracking data
employee_data = {}
log_data1 = []  # Employee log data
log_data2 = []  # General detections log data

# Function to save logs to separate CSV files
def save_to_csv(filename, log_data, columns):
    if log_data:
        df = pd.DataFrame(log_data, columns=columns)
        df.to_csv(filename, index=False)
        print(f"CSV file saved as '{filename}'")

# Function to process the video frames and detect behaviors
def process_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Selected video file: {file_path}")
        cap = cv2.VideoCapture(file_path)

        def process_and_display_frame():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (640, 640))
                height, width = frame_resized.shape[:2]
                annotator = Annotator(frame_resized)
                all_detections_employee = []
                all_detections_person = []

                # Split the frame into 4 quadrants and process each independently
                quadrants = [
                    frame_resized[0:height // 2, 0:width // 2],
                    frame_resized[0:height // 2, width // 2:width],
                    frame_resized[height // 2:height, 0:width // 2],
                    frame_resized[height // 2:height, width // 2:width]
                ]

                for i, quad in enumerate(quadrants):
                    quad_resized = cv2.resize(quad, (640, 640))

                    # Employee detection using the first model
                    results_employee = model_employee.predict(quad)
                    for result in results_employee:
                        boxes = result.boxes
                        for box in boxes:
                            b = box.xyxy[0]  # box coordinates
                            c = int(box.cls) if box.cls is not None else -1
                            color = model_colors[model_employee]

                            # Adjust bounding box coordinates for quadrant offset
                            offset_x = (i % 2) * (width // 2)
                            offset_y = (i // 2) * (height // 2)
                            adjusted_b = [b[0].item() + offset_x, b[1].item() + offset_y, 
                                          b[2].item() + offset_x, b[3].item() + offset_y]
                            annotator.box_label(adjusted_b,
                                                f"{class_names_employee[c]} ({box.conf.item():.2f})",
                                                color=color)
                            confidence = box.conf.item() if box.conf is not None else 1.0
                            if confidence >= 0.05:
                                all_detections_employee.append((adjusted_b[0], adjusted_b[1], adjusted_b[2],
                                                                 adjusted_b[3], confidence, c))

                    # General detection using YOLOv8n model
                    results_general = model_general.predict(quad, classes=[0])
                    for result in results_general:
                        boxes = result.boxes
                        for box in boxes:
                            b = box.xyxy[0]
                            c = int(box.cls) if box.cls is not None else -1
                            color = model_colors[model_general]
                            offset_x = (i % 2) * (width // 2)
                            offset_y = (i // 2) * (height // 2)
                            adjusted_b = [b[0].item() + offset_x, b[1].item() + offset_y,
                                          b[2].item() + offset_x, b[3].item() + offset_y]
                            annotator.box_label(adjusted_b,
                                                f"{class_names_general[c]} ({box.conf.item():.2f})",
                                                color=color)
                            confidence = box.conf.item() if box.conf is not None else 1.0
                            if confidence >= 0.4:
                                all_detections_person.append((adjusted_b[0], adjusted_b[1], adjusted_b[2],
                                                              adjusted_b[3], confidence, c))

                # Process employee detections
                if all_detections_employee:
                    bbs_employee = [(det[:4], det[4], det[5]) for det in all_detections_employee]
                    tracks_employee = behavior_tracker.update_tracks(bbs_employee, frame=frame_resized)

                    for track in tracks_employee:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id

                        if track_id not in employee_data:
                            employee_data[track_id] = {"points": 100}

                        bbox = track.to_ltrb()
                        detection_class = track.det_class

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_data1.append([timestamp, track_id,
                                          class_names_employee[detection_class] if detection_class >= 0 else "Unknown",
                                          list(bbox)])

                # Process general detections
                if all_detections_person:
                    bbs_person = [(det[:4], det[4], det[5]) for det in all_detections_person]
                    tracks_person = person_tracker.update_tracks(bbs_person, frame=frame_resized)

                    for track in tracks_person:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id

                        bbox = track.to_ltrb()
                        detection_class = track.det_class

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_data2.append([timestamp, track_id,
                                          class_names_general[detection_class] if detection_class >= 0 else "Unknown",
                                          list(bbox)])

                annotated_frame = annotator.result()
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.thumbnail((640, 640))
                photo_image = ImageTk.PhotoImage(frame_pil)
                canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
                canvas.image = photo_image

                fps = cap.get(cv2.CAP_PROP_FPS)
                delay = int(1000 / fps) if fps > 0 else 100
                window.after(delay, process_and_display_frame)
            else:
                cap.release()
                save_to_csv(r"C:\Users\geeeg\OneDrive\Documents\employee\employee_behavior_log.csv", log_data1, ["Timestamp", "Employee_ID", "Behavior", "BBox"])
                save_to_csv(r"C:\Users\geeeg\OneDrive\Documents\employee\general_behavior_log.csv", log_data2, ["Timestamp", "Person_ID", "Class", "BBox"])
                postprocess_behavior_mapping(r"C:\Users\geeeg\OneDrive\Documents\employee\general_behavior_log.csv",r"C:\Users\geeeg\OneDrive\Documents\employee\employee_behavior_log.csv", r"C:\Users\geeeg\OneDrive\Documents\employee\final_behavior_mapping.csv")

        process_and_display_frame()

# Define behavior ratings based on employee behaviors
BEHAVIOR_RATINGS = {
    'At-Desk-NotWorking': -5,
    'At-Desk-Working': 10,
    'Sleeping': -5,
    'Standing-NotWorking': -3,
    'Standing-Working': 10,
    'Walking': 0,
}

# Postprocessing function to map behaviors to persons with ratings
# Postprocessing function to map behaviors to persons with ratings
def postprocess_behavior_mapping(person_csv, behavior_csv, output_csv):
    person_df = pd.read_csv(person_csv)
    behavior_df = pd.read_csv(behavior_csv)

    person_df['BBox'] = person_df['BBox'].apply(eval)
    behavior_df['BBox'] = behavior_df['BBox'].apply(eval)

    results = []

    for _, behavior_row in behavior_df.iterrows():
        behavior_timestamp = behavior_row['Timestamp']
        behavior_bbox = behavior_row['BBox']
        behavior_name = behavior_row['Behavior']

        persons_at_time = person_df[person_df['Timestamp'] == behavior_timestamp]
        if persons_at_time.empty:
            continue

        closest_person_id = None
        min_distance = float('inf')

        for _, person_row in persons_at_time.iterrows():
            person_id = person_row['Person_ID']
            person_bbox = person_row['BBox']

            # Calculate the center of the bounding boxes
            behavior_center = ((behavior_bbox[0] + behavior_bbox[2]) / 2, (behavior_bbox[1] + behavior_bbox[3]) / 2)
            person_center = ((person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2)

            # Calculate the distance between the centers
            distance = np.sqrt((behavior_center[0] - person_center[0])**2 + (behavior_center[1] - person_center[1])**2)
            if distance < min_distance:
                closest_person_id = person_id
                min_distance = distance

        # Get the rating for the detected behavior
        rating = BEHAVIOR_RATINGS.get(behavior_name, 0)  # Default to 0 if not found

        results.append([behavior_timestamp, closest_person_id, behavior_name, rating])

    # Create a DataFrame from results
    final_df = pd.DataFrame(results, columns=["Timestamp", "Person_ID", "Behavior", "Rating"])

    # Remap Person_IDs to start from 1
    unique_person_ids = final_df['Person_ID'].unique()
    id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_person_ids, start=1)}
    
    # Apply the mapping to remap Person_IDs
    final_df['Person_ID'] = final_df['Person_ID'].map(id_mapping)

    # Save the final DataFrame to CSV
    final_df.to_csv(output_csv, index=False)
    print(f"Final mapping saved to '{output_csv}'")

# GUI setup
window = tk.Tk()
window.title("YOLO Video Processing")
canvas = tk.Canvas(window, width=640, height=640)
canvas.pack()
browse_button = tk.Button(window, text="Browse and Process Video", command=process_video)
browse_button.pack()
window.mainloop()
