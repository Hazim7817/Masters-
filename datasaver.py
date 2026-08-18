#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:57:11 2024

@author: dmv
"""

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pandas as pd
import os
import csv

#Use feather to efficiently save files to disk that can later be read in by R
class datasaver:
    def __init__(self,directoryname):
        self.directory = "./"+directoryname
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
    
    #This looks really slow, but takes less than a millisecond on my mediocre laptop
    def savegrid(self,grid,name):
        saveabledata = pa.array(np.ndarray.flatten(grid))
        schema = pa.schema([pa.field('nums', saveabledata.type)])
    
        with pa.OSFile(self.directory+"/"+name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema=schema) as writer:
                batch = pa.record_batch([saveabledata], schema=schema)
                writer.write(batch)
    
    def savecells(self,cells,name):
        cellxlocs = []
        cellylocs = []
        cellids = []
        for cell in cells:
            cellxlocs.append(cell.xlocation)
            cellylocs.append(cell.ylocation)
            cellids.append(cell.id)
            #save how much it produced 

        saveabledata = pa.Table.from_arrays([cellxlocs, cellylocs,cellids], names=["xloc","yloc","id"])
        feather.write_feather(saveabledata, self.directory+"/"+name)
    
    def savecellsextended(self,cells,name):
        cellxlocs = []
        cellylocs = []
        cellids = []
        cellproduced = []
        cellconsumed = []
        cellaccumulatedproduce = []
        for cell in cells:
            cellxlocs.append(cell.xlocation)
            cellylocs.append(cell.ylocation)
            cellids.append(cell.id)
            cellproduced.append(cell.producedthisstep)
            cellconsumed.append(cell.consumedthisstep)
            cellaccumulatedproduce.append(cell.accumulatedproduction)

        #RESET THE DATA HEre
        saveabledata = pa.Table.from_arrays([cellxlocs, cellylocs,cellids,cellproduced,cellconsumed,cellaccumulatedproduce], names=["xloc","yloc","id","prod","cons","totalprod"])
        feather.write_feather(saveabledata, self.directory+"/"+name)
        
    #Not usually used
    def savecellscsv(self,cells,name,time):
        with open(self.directory+"/"+name+'.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            for cell in cells:
                xloc = cell.xlocation
                yloc = cell.ylocation
                cellid = cell.id
                writer.writerow([time,cellid,xloc,yloc])
        
 
    
        
        


