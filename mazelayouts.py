#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:45:56 2024

@author: dmv
"""
import numpy as np

#Various shapes of walls to use in simulations
#If there's no walls around the outside, diffusion will cause a lot of it to empty out

def makewalls(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[-1, :] = 1  # block
    
    grid_walls[0:int(0.8*x),int(0.2*x):int(0.25*x)] = 1;
    grid_walls[int(0.2*x):int(1*x),int(0.4*x):int(0.45*x)] = 1;
    grid_walls[0:int(0.8*x),int(0.6*x):int(0.65*x)] = 1;
    grid_walls[int(0.2*x):int(1*x),int(0.8*x):int(0.85*x)] = 1;
    return grid_walls

def makeemptycontainer(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[-1, :] = 1  # block
    return grid_walls

def makeemptydish(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, y-2:y] = 1  # block
    grid_walls[0:2, :] = 1  # block
    grid_walls[:,0:2] = 1  # block
    grid_walls[x-2:x, :] = 1  # block
    return grid_walls

def makesixtchamberdish(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, y-2:y] = 1  # block
    grid_walls[0:2, :] = 1  # block
    grid_walls[:,0:2] = 1  # block
    grid_walls[x-2:x, :] = 1  # block
    for xloc in range(0,x):
        for yloc in range(0,y):
            if ((xloc%40)>20) and ((yloc%40)>20):
                grid_walls[xloc,yloc]=1
    return grid_walls

def makeopensite(x,y):
    grid_walls = np.zeros((x, y))
    return grid_walls

def makebridgerelative(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[:,1] = 1  # block
    grid_walls[-1, :] = 1  # block
    
    grid_walls[0:int(0.166*x),int(0.33*y):int(0.66*y)] = 1;
    grid_walls[int(0.844*x):x,int(0.33*y):int(0.66*y)] = 1;
    return grid_walls


def makebridgesmall(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[:,1] = 1  # block
    grid_walls[-1, :] = 1  # block
    
    grid_walls[0:int(50),100:300] = 1;
    grid_walls[int(250):x,100:300] = 1;
    return grid_walls

def makebridge(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[:,1] = 1  # block
    grid_walls[-1, :] = 1  # block
    
    grid_walls[0:int(0.166*x),250:650] = 1;
    grid_walls[int(0.844*x):x,250:650] = 1;
    return grid_walls

def makemultiwall(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[:,1] = 1  # block
    grid_walls[-1, :] = 1  # block
    #grid_walls[0:int(0.166*x),int(0.33*y):int(0.66*y)] = 1;
    #grid_walls[int(0.844*x):x,int(0.33*y):int(0.66*y)] = 1;
    grid_walls[50:60,0:y-50]=1
    grid_walls[70:80,0:y-50]=1
    
    return grid_walls

def makebloodvessel(x,y):
    grid_walls = np.zeros((x, y))
    grid_walls[:, -1] = 1  # block
    grid_walls[1, :] = 1  # block
    grid_walls[-1, :] = 1  # block
    
    grid_walls[int(0.1*x):int(0.15*x),0:int(0.2*y)] = 1;
    grid_walls[int(0.1*x):int(0.15*x),int(0.2*y)+3:int(0.4*y)] = 1;
    grid_walls[int(0.1*x):int(0.15*x),int(0.4*y)+3:int(0.6*y)] = 1;
    grid_walls[int(0.1*x):int(0.15*x),int(0.6*y)+3:int(0.8*y)] = 1;
    grid_walls[int(0.1*x):int(0.15*x),int(0.8*y)+3:y] = 1;
    #grid_walls[int(0.6*x):x,int(0.2*y):int(0.8*y)] = 1;
    return grid_walls