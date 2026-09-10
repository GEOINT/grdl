"""
Modality Injectors for geometric shapes.

Author
------
GRDX Team
"""

import numpy as np
from typing import Any, Dict, Tuple
from grdl.synthetic.contracts import ImageInjectorBase, PhysicalObject
from grdl.image_processing.versioning import processor_tags
from grdl.vocabulary import ImageModality
from grdl.image_processing.detection.fields import Fields

@processor_tags(modalities=[ImageModality.EO])
class GeometricEOInjector(ImageInjectorBase):
    @property
    def injector_name(self) -> str:
        return "geometric_eo_injector"

    def inject(self, base_chip: np.ndarray, obj: PhysicalObject, position_xy: Tuple[float, float], sensor_perspective: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = int(position_xy[0]), int(position_xy[1])
        w, h = int(obj.dimensions.get("width", 10)), int(obj.dimensions.get("height", 10))
        chip = base_chip.copy()
        mask = np.zeros(chip.shape[:2], dtype=bool)
        
        color = (255, 0, 0)
        if obj.materials.get(Fields.trait.MATERIAL) == "metal":
            color = (200, 200, 200)
            
        x_min, x_max = max(0, x - w//2), min(chip.shape[1], x + w//2)
        y_min, y_max = max(0, y - h//2), min(chip.shape[0], y + h//2)
        
        if len(chip.shape) == 3:
            chip[y_min:y_max, x_min:x_max] = color
        mask[y_min:y_max, x_min:x_max] = True
        return chip, mask

@processor_tags(modalities=[ImageModality.SAR])
class GeometricSARInjector(ImageInjectorBase):
    @property
    def injector_name(self) -> str:
        return "geometric_sar_injector"

    def inject(self, base_chip: np.ndarray, obj: PhysicalObject, position_xy: Tuple[float, float], sensor_perspective: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = int(position_xy[0]), int(position_xy[1])
        w, h = int(obj.dimensions.get("width", 10)), int(obj.dimensions.get("height", 10))
        chip = base_chip.copy()
        mask = np.zeros(chip.shape[:2], dtype=bool)
        
        rcs = obj.materials.get(Fields.physical.RCS, 0.5)
        
        x_min, x_max = max(0, x - w//2), min(chip.shape[1], x + w//2)
        y_min, y_max = max(0, y - h//2), min(chip.shape[0], y + h//2)
        
        chip[y_min:y_max, x_min:x_max] += rcs
        mask[y_min:y_max, x_min:x_max] = True
        return chip, mask
