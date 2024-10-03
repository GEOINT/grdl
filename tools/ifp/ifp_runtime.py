# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:28:04 2024

@author: User
"""

import json
from utils.IFP_Proc import IFP_Proc
from utils.compress_phd_data import compress_phd_data


input_info = json.load( open( "run_file.json"))
index = 0

run_dict_input = { 'file_name':input_info['file_name'][index],
                   'data_path':input_info['data_path'],
                   'range_compression':input_info['range_compression'],
                   'azimuth_compression':input_info['azimuth_compression'],
                   'Grid_Inscription':input_info['Grid_Inscription'],
                   'IFP_Projection':input_info['IFP_Projection']}

ifp = IFP_Proc(run_dict_input)
ifp.populate_metadata()
ifp.populate_geometry(plot_collection_geo = True)

ifp.populate_polargrid()

#ifp.perform_rma()

ifp.perform_pfa()

ifp.save_sicd()
ifp.plot_image()


a = 1