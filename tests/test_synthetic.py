"""
Unit tests for grdl.synthetic components.
"""
import pytest
import numpy as np
from grdl.synthetic.generators import GeometricShapeGenerator
from grdl.synthetic.injectors import GeometricEOInjector, GeometricSARInjector
from grdl.image_processing.detection.fields import Fields

from grdl.vocabulary import ImageModality

def test_geometric_shape_generator():
    gen = GeometricShapeGenerator()
    params = {"shape": "rectangle", "width": 20, "height": 30}
    obj = gen.generate_object(params)
    assert obj.geometry_type == "rectangle"
    assert obj.dimensions["width"] == 20
    assert obj.materials[Fields.trait.MATERIAL] == "generic"

def test_eo_injector():
    injector = GeometricEOInjector()
    assert ImageModality.EO in injector.__processor_tags__.get("modalities", [])
    
    gen = GeometricShapeGenerator()
    obj = gen.generate_object({"width": 10, "height": 10})
    
    chip = np.zeros((100, 100, 3), dtype=np.uint8)
    out_chip, mask = injector.inject(chip, obj, (50, 50), {})
    
    assert mask.any()
    assert (out_chip[50, 50] == [255, 0, 0]).all()

def test_sar_injector():
    injector = GeometricSARInjector()
    assert ImageModality.SAR in injector.__processor_tags__.get("modalities", [])
    
    gen = GeometricShapeGenerator()
    obj = gen.generate_object({"width": 10, "height": 10})
    
    chip = np.zeros((100, 100), dtype=np.complex64)
    out_chip, mask = injector.inject(chip, obj, (50, 50), {})
    
    assert mask.any()
    assert out_chip[50, 50].real > 0
