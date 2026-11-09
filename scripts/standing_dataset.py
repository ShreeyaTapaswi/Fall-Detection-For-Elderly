from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["standing", "falling", "fallen"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class StandingDatasetPair:
    video_path: Path
    annotation_path: Path
    source_name: str


def parse_cvat_frame_labels(xml_path: Path) -> dict[int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_labels: dict[int, int] = {}
    for track in root.findall(".//track"):
        label = (track.get("label") or "").strip().lower()
        if label not in CLASS_TO_INDEX:
            continue

        class_id = CLASS_TO_INDEX[label]
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame_labels[int(box.get("frame"))] = class_id

    return frame_labels


def discover_standing_dataset_pairs(project_root: Path) -> list[StandingDatasetPair]:
    videos_dir = project_root / "standing dataset for training" / "Standing"
    annotations_dir = project_root / "standing dataset for training" / "annotations"
    pairs: list[StandingDatasetPair] = []

    if not videos_dir.exists() or not annotations_dir.exists():
        return pairs

    for xml_path in sorted(annotations_dir.glob("*.xml")):
        video_path = videos_dir / f"{xml_path.stem}.mp4"
        if video_path.exists():
            pairs.append(
                StandingDatasetPair(
                    video_path=video_path,
                    annotation_path=xml_path,
                    source_name=xml_path.stem,
                )
            )

    return pairs
