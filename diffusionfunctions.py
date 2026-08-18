#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:07:30 2024

@author: dmv
"""

import numba
import numpy as np
import math

#This is used for one step after initialization, as dufort-frankel requires two steps of initial conditions
@numba.jit(nopython=True)
def step_diffusion_forwardeuler(grid,grid_walls,D):
    grid_new = np.copy(grid)  # Create a copy to store the updated values
    
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            #dissect into components
            if grid_walls[i,j]==1:
               continue
            openj = 0
            updiffuse = 0
            downdiffuse = 0
            #up
            if(grid_walls[i,j+1]<1): #If there's no wall here
                openj+=1
                updiffuse = grid[i, j+1];
            #down
            if(grid_walls[i,j-1]<1): #If there's no wall here
                openj+=1
                downdiffuse = grid[i, j-1]
            openi = 0
            leftdiffuse = 0
            rightdiffuse = 0
            #left
            if(grid_walls[i+1,j]<1): #If there's no wall here
                openi+=1
                leftdiffuse = grid[i+1, j]
            #right
            if(grid_walls[i-1,j]<1): #If there's no wall here
                openi+=1
                rightdiffuse = grid[i-1, j]
            totalopen = openj + openi
            grid_new[i, j] = grid[i, j] + D * (leftdiffuse + rightdiffuse+updiffuse+ downdiffuse- totalopen*grid[i, j])
    return(grid_new)

#This is used to calculate diffusion for every timestep
@numba.jit(nopython=True)
def step_diffusion_dufortfrankel(grid,old_grid,grid_walls,D):
    grid_new = np.copy(grid)  # Create a copy to store the updated values
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            if grid_walls[i,j]==1:
               continue
            #dissect into components

            updiffuse = 0
            downdiffuse = 0
            #up
            if(grid_walls[i,j+1]<1): #If there's no wall here
                updiffuse = grid[i, j+1];
            else:
                updiffuse = grid[i, j];
            #down
            if(grid_walls[i,j-1]<1): #If there's no wall here
                downdiffuse = grid[i, j-1]
            else:
                downdiffuse = grid[i, j]
            jdiffuse = (1/(1+4*D))*2*D*(updiffuse + downdiffuse)
            

            leftdiffuse = 0
            rightdiffuse = 0
            #left
            if(grid_walls[i+1,j]<1): #If there's no wall here
                leftdiffuse = grid[i+1, j]
            else:
                leftdiffuse = grid[i, j]
            
            #right
            if(grid_walls[i-1,j]<1): #If there's no wall here
                rightdiffuse = grid[i-1, j]
            else:
                rightdiffuse = grid[i, j]
                
            idiffuse = (1/(1+4*D))*2*D*(leftdiffuse + rightdiffuse)
            
            oldcontribution = (1/(1+4*D))*(1-4*D)*old_grid[i,j]
            grid_new[i,j] = oldcontribution+jdiffuse+idiffuse
            #grid_new[i, j] = (1/(1+totalopen*D))*((1-totalopen*D)*old_grid[i,j]+jdiffuse+idiffuse)
     
            #grid_new[i, j] = (1/(1+4*D))*((1-4*D)*old_grid[i,j]+2*D*(grid[i, j+1] + grid[i, j-1])+2*D*(grid[i+1, j] + grid[i-1, j]))
    #print(cuda.current_context())
    return(grid_new,grid)

@numba.jit(nopython=True)
def step_diffusion_dufortfrankelalternatewall(grid,old_grid,grid_walls,D):
    grid_new = np.copy(grid)  # Create a copy to store the updated values
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            if grid_walls[i,j]==1:
               continue
            #dissect into components

            updiffuse = 0
            downdiffuse = 0
            #up
            if(grid_walls[i,j+1]<1): #If there's no wall here
                updiffuse = grid[i, j+1];
            else:
                updiffuse = old_grid[i, j];
            #down
            if(grid_walls[i,j-1]<1): #If there's no wall here
                downdiffuse = grid[i, j-1]
            else:
                downdiffuse = old_grid[i, j]
            jdiffuse = (1/(1+4*D))*2*D*(updiffuse + downdiffuse)
            

            leftdiffuse = 0
            rightdiffuse = 0
            #left
            if(grid_walls[i+1,j]<1): #If there's no wall here
                leftdiffuse = grid[i+1, j]
            else:
                leftdiffuse = old_grid[i, j]
            
            #right
            if(grid_walls[i-1,j]<1): #If there's no wall here
                rightdiffuse = grid[i-1, j]
            else:
                rightdiffuse = old_grid[i, j]
                
            idiffuse = (1/(1+4*D))*2*D*(leftdiffuse + rightdiffuse)
            
            oldcontribution = (1/(1+4*D))*(1-4*D)*old_grid[i,j]
            grid_new[i,j] = oldcontribution+jdiffuse+idiffuse
            #grid_new[i, j] = (1/(1+totalopen*D))*((1-totalopen*D)*old_grid[i,j]+jdiffuse+idiffuse)
     
            #grid_new[i, j] = (1/(1+4*D))*((1-4*D)*old_grid[i,j]+2*D*(grid[i, j+1] + grid[i, j-1])+2*D*(grid[i+1, j] + grid[i-1, j]))
    #print(cuda.current_context())
    return(grid_new,grid)

@numba.jit(nopython=True)
def step_decay(grid,decayrate):
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            grid[i,j] = grid[i,j]* math.exp(-1/decayrate)
    return(grid)