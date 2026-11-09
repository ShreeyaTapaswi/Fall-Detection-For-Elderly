import xml.etree.ElementTree as ET
import os
from pathlib import Path

def convert_cvat_xml_to_yolo(xml_file, output_dir, video_id, img_width=1920, img_height=1080):
    """
    Parse CVAT XML and generate YOLO .txt files.
    Naming: v{video_id}_f{frame:06d}.txt
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    label_map = {
        "Sleeping": 0,
        "Falling": 1,
        "Fallen": 2
    }
    
    # Track counts for debugging
    counts = {k: 0 for k in label_map.keys()}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for track in root.findall(".//track"):
        label = track.get("label")
        if label not in label_map:
            continue
            
        class_id = label_map[label]
        
        for box in track.findall("box"):
            frame_idx = int(box.get("frame"))
            outside = box.get("outside")
            
            if outside == "1":
                continue
                
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            
            # Normalize coordinates
            w = (xbr - xtl) / img_width
            h = (ybr - ytl) / img_height
            x_center = (xtl + (xbr - xtl) / 2) / img_width
            y_center = (ytl + (ybr - ytl) / 2) / img_height
            
            # Constraints: ensure they are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            output_file = os.path.join(output_dir, f"v{video_id}_f{frame_idx:06d}.txt")
            
            # Append if frame has multiple detections (though in this case tracks are probably unique per frame)
            with open(output_file, "a") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            
            counts[label] += 1
            
    print(f"✅ Converted {xml_file} (Video ID: {video_id})")
    for lbl, count in counts.items():
        print(f"   - {lbl}: {count} boxes")

if __name__ == "__main__":
    base_dir = Path("c:/Users/ADMIN/Documents/Fall_Detection_2")
    output_labels_dir = base_dir / "dataset/labels/train" # Match images/train
    
    # Process both XMLs
    convert_cvat_xml_to_yolo(str(base_dir / "annotations_1.xml"), str(output_labels_dir), video_id=1)
    convert_cvat_xml_to_yolo(str(base_dir / "annotations_2.xml"), str(output_labels_dir), video_id=2)
