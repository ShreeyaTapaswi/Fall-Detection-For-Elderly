import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import os

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    label_map = {"Sleeping": 0, "Falling": 1, "Fallen": 2}
    frame_labels = {}
    
    for track in root.findall(".//track"):
        label = track.get("label")
        if label not in label_map:
            continue
        
        class_id = label_map[label]
        for box in track.findall("box"):
            frame_idx = int(box.get("frame"))
            outside = box.get("outside")
            if outside == "0":
                frame_labels[frame_idx] = class_id
                
    return frame_labels

def extract_sequences(video_path, xml_path, model_path, seq_len=30):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    frame_labels = parse_annotations(xml_path)
    
    sequences = []
    labels = []
    
    current_keypoints_seq = []
    current_label_seq = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting keypoints from {video_path}...")
    
    for f_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get ground truth label for this frame
        label = frame_labels.get(f_idx, -1)
        if label == -1:
            current_keypoints_seq = [] # Reset on gap
            continue
            
        # Run pose estimation
        results = model(frame, verbose=False)[0]
        
        if len(results.keypoints) > 0:
            # Get 17 keypoints (x, y, conf) -> 51 values
            kpts = results.keypoints[0].data.cpu().numpy()[0] # (17, 3)
            kpts_flat = kpts.flatten() # (51,)
            
            current_keypoints_seq.append(kpts_flat)
            current_label_seq.append(label)
            
            if len(current_keypoints_seq) == seq_len:
                sequences.append(np.array(current_keypoints_seq))
                # Label of sequence is the label of the last frame (or majority)
                labels.append(label)
                
                # Sliding window with overlap
                current_keypoints_seq = current_keypoints_seq[1:]
                current_label_seq = current_label_seq[1:]
        else:
            current_keypoints_seq = [] # Reset on detection failure
            
    cap.release()
    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    base_dir = Path("c:/Users/ADMIN/Documents/Fall_Detection_2")
    # 1. Path to pose model (Using yolov8n-pose as fallback for demo)
    model_path = 'yolov8n-pose.pt'
    
    all_seqs = []
    all_labs = []
    
    dataset_configs = [
        {"v": base_dir / "train_video_1.mp4", "x": base_dir / "annotations_1.xml"},
        {"v": base_dir / "train_video_2.mp4", "x": base_dir / "annotations_2.xml"}
    ]
    
    for config in dataset_configs:
        s, l = extract_sequences(config["v"], config["x"], model_path)
        if len(s) > 0:
            all_seqs.append(s)
            all_labs.append(l)
            
    if all_seqs:
        X = np.concatenate(all_seqs, axis=0)
        y = np.concatenate(all_labs, axis=0)
        
        np.save("X_sequences.npy", X)
        np.save("y_labels.npy", y)
        print(f"✅ Extracted {len(X)} sequences. Saved to X_sequences.npy and y_labels.npy")
    else:
        print("❌ No sequences extracted.")
