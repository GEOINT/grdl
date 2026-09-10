"""
Geometric Shape Generators for synthetic data.

Author
------
GRDX Team
"""

from typing import Any, Dict
from grdl.synthetic.contracts import ObjectGeneratorBase, PhysicalObject
from grdl.image_processing.detection.fields import Fields

class GeometricShapeGenerator(ObjectGeneratorBase):
    @property
    def generator_name(self) -> str:
        return "geometric_shape"

    def generate_object(self, params: Dict[str, Any]) -> PhysicalObject:
        shape_type = params.get("shape", "rectangle")
        width = params.get("width", 10.0)
        height = params.get("height", 10.0)
        
        materials = {
            Fields.trait.MATERIAL: params.get("material", "generic"),
            Fields.physical.RCS: params.get("rcs", 0.5)
        }
        
        return PhysicalObject(
            geometry_type=shape_type,
            dimensions={"width": width, "height": height},
            materials=materials,
            metadata={}
        )
