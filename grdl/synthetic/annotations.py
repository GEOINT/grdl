"""
Annotation writers for synthetic data.

Author
------
GRDX Team
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from grdl.synthetic.contracts import AnnotationWriterBase

class GeoJSONWriter(AnnotationWriterBase):
    @property
    def format_name(self) -> str:
        return "geojson"

    def write_annotation(self, output_dir: str, base_filename: str, bounding_boxes: List[Tuple[float, float, float, float]], masks: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        features = []
        for bbox, meta in zip(bounding_boxes, metadata):
            xmin, ymin, xmax, ymax = bbox
            feature = {
                "type": "Feature",
                "properties": meta,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]
                }
            }
            features.append(feature)
        
        fc = {"type": "FeatureCollection", "features": features}
        out_path = Path(output_dir) / f"{base_filename}.geojson"
        with open(out_path, "w") as f:
            json.dump(fc, f, indent=2)


class COCOWriter(AnnotationWriterBase):
    @property
    def format_name(self) -> str:
        return "coco"

    def write_annotation(self, output_dir: str, base_filename: str, bounding_boxes: List[Tuple[float, float, float, float]], masks: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        annotations = []
        for i, (bbox, meta) in enumerate(zip(bounding_boxes, metadata)):
            xmin, ymin, xmax, ymax = bbox
            annotations.append({
                "id": i,
                "image_id": base_filename,
                "category_id": 1,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "attributes": meta
            })
        out_path = Path(output_dir) / f"{base_filename}_coco.json"
        with open(out_path, "w") as f:
            json.dump({"annotations": annotations}, f, indent=2)


class YOLOWriter(AnnotationWriterBase):
    @property
    def format_name(self) -> str:
        return "yolo"

    def write_annotation(self, output_dir: str, base_filename: str, bounding_boxes: List[Tuple[float, float, float, float]], masks: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        lines = []
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            # Simplified for stub, assuming 256x256 image chip size
            x_center = ((xmin + xmax) / 2) / 256.0
            y_center = ((ymin + ymax) / 2) / 256.0
            width = (xmax - xmin) / 256.0
            height = (ymax - ymin) / 256.0
            lines.append(f"0 {x_center} {y_center} {width} {height}")
            
        out_path = Path(output_dir) / f"{base_filename}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
