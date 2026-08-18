#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:07:30 2024

@author: dmv
"""

import numba
import numpy as np
import math
#@cuda.jit
#This is used for one step after initialization, as dufort-frankel requires two steps of initial conditions
@numba.jit(nopython=True)
def step_diffusion_forwardeuler(grid,grid_walls,D):
    grid_new = np.copy(grid)  # Create a copy to store the updated values
    if D>0.25:
        D=0.25
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            #dissect into components
            if grid_walls[i,j]==1:
               continue
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
            grid_new[i, j] = grid[i, j] + D * (leftdiffuse + rightdiffuse+updiffuse+ downdiffuse- 4*grid[i, j])
            #print( grid_new[i, j])
            if grid_new[i, j]>100000:
                print("whoops")
                print(grid[i, j])
                print(i)
                print(j)
                print(leftdiffuse)
                print(rightdiffuse)
                print(updiffuse)
                print(downdiffuse)

    return(grid_new,grid)

#This is used to calculate diffusion for every timestep after initialisation
@numba.jit(nopython=True)
def step_diffusion_dufortfrankel(grid,old_grid,grid_walls,D):
    grid_new = np.copy(grid)  # Create a copy to store the updated values
    #precalculate a few things
    modifier = ((1-2*D)/(1+2*D))
    modifier2 = (D/(1+2*D))
    #Get into the diffusion loop
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            #Ignore this point if we're inside a wall
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
           
            #Combine components into a new value for this point
            grid_new[i,j] = modifier*old_grid[i,j]+modifier2*(rightdiffuse+leftdiffuse+updiffuse+downdiffuse)

    return(grid_new,grid)

#Have a specific break down with a half-life on every location in the grid
@numba.jit(nopython=True)
def step_decay(grid,halflife):
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            grid[i,j] = grid[i,j]* math.exp(-1/halflife)
    return(grid)

#Same effect as decay function above, but returns the amount broken down
#Can be useful if you want a ligand to break down into a different ligand
@numba.jit(nopython=True)
def step_decay_withreturn(grid,halflife):
    Nx = grid.shape[0]
    Ny = grid.shape[1]
    decaysites = np.zeros(Nx*Ny,dtype=np.int64)
    decayvalues = np.zeros(Nx*Ny)
    sitecount = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            pastvalue = grid[i,j]
            if pastvalue>0:
               # print(pastvalue)
                grid[i,j] = grid[i,j]* math.exp(-1/halflife)
                decaysites[sitecount] = j*Nx+i
                decayvalues[sitecount] = pastvalue - grid[i,j]
                sitecount +=1
    return(grid,decaysites[0:sitecount],decayvalues[0:sitecount])

#Remove some ligand with a df in a specific area
#Used to represent consumption or extracellular breakdown around cells
@numba.jit(nopython=True)
def step_consume(grid,coveredsquares,consumptiondf,consumptionvmax):
    totalconsumed = 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        localconcentration = grid[i,j]
        if localconcentration>0:
            #grid[i,j]=grid[i,j]*(localconcentration/(localconcentration+consumptiondf))
            grid[i,j]=grid[i,j]-((localconcentration*consumptionvmax)/(localconcentration+consumptiondf)) 
            if grid[i,j]<0:
                grid[i,j]=0
            totalconsumed += (localconcentration-grid[i,j])
    return grid, totalconsumed

@numba.jit(nopython=True)
def step_consume_linear(grid,coveredsquares,consumptiondf):
    totalconsumed = 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        localconcentration = grid[i,j]
        if localconcentration>0:
            #grid[i,j]=grid[i,j]*(localconcentration/(localconcentration+consumptiondf))
            grid[i,j]=grid[i,j]-((localconcentration)/(localconcentration+consumptiondf))*0.005 
            if grid[i,j]<0:
                grid[i,j]=0
            totalconsumed += (localconcentration-grid[i,j])
    return grid, totalconsumed

@numba.jit(nopython=True)
def step_consume_proportional(grid,coveredsquares):
    totalconsumed = 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        localconcentration = grid[i,j]
        if localconcentration>0:
            #grid[i,j]=grid[i,j]*(localconcentration/(localconcentration+consumptiondf))
            grid[i,j]=grid[i,j]-localconcentration*0.05
            if grid[i,j]<0:
                grid[i,j]=0
            totalconsumed += (localconcentration-grid[i,j])
    return grid, totalconsumed

#As above, but returns how much has been consumed/broken down
#Can be used to have a ligand break down into another ligand without moving
@numba.jit(nopython=True)
def step_consume_returnsites(grid,coveredsquares,consumptiondf,consumptionvmax):
    Nx = grid.shape[0]
    consumevalues = np.zeros(len(coveredsquares))
    sitecount=0
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        localconcentration = grid[i,j]
        if localconcentration>0:
            grid[i,j]=grid[i,j]-((localconcentration*consumptionvmax)/(localconcentration+consumptiondf)) 
            if grid[i,j]<0:
                #print("Oopsie! negative value")
                #print(grid[i,j])
                grid[i,j]=0
            consumevalues[sitecount] = (localconcentration-grid[i,j])
        sitecount +=1
    return grid, consumevalues

#Produce a ligand in a specific area, usually under a cell
@numba.jit(nopython=True)
def step_produce(grid,coveredsquares,production):
    Nx = grid.shape[0]
    sitecount = 0
    for site in coveredsquares:
        i = site%Nx
        j = math.floor(site / Nx)
        addedconcentration = (production[sitecount])
        grid[i,j] += addedconcentration
        sitecount +=1
    return grid

#Sense the amount of ligand locally
@numba.jit(nopython=True)
def sense_jitted_kd(grid,coveredsquares,kd):
    totalsensed = 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        localconcentration = grid[i,j]
        if localconcentration>0:
            totalsensed += localconcentration- (grid[i,j]*(localconcentration/(localconcentration+kd)))
    return totalsensed

#Sense the amount of ligand locally with a kd, as if consuming
@numba.jit(nopython=True)
def sense_jitted(grid,coveredsquares):
    totalsensed= 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        totalsensed += grid[i,j]
    return totalsensed
class ligand:
    def __init__(self,xsize,ysize,halflife,diffuserate,grid_walls,name):
        self.xsize = xsize
        self.ysize = ysize
        self.halflife = halflife
        self.diffuserate = diffuserate
        self.grid_prev = np.zeros((xsize, ysize))  # initial condition (e.g., zero everywhere), may be set to something else
        self.grid = np.zeros((xsize, ysize)) #Will be set by the first euler step
        self.name = name
    
    #These methods are mostly wrappers for functions that have been sped up with numba
    
    #Done as a first step of diffusion after initialisation
    def diffuse_euler_init(self,grid_walls):
        self.grid, self.grid_prev  = step_diffusion_forwardeuler(self.grid_prev, grid_walls, self.diffuserate)
    
    #Done as a first step of diffusion after initialisation
    def diffuse_euler(self,grid_walls):
        self.grid, self.grid_prev  = step_diffusion_forwardeuler(self.grid, grid_walls, self.diffuserate)
        
    #Used for diffusion in later steps
    def diffuse_dufort(self,grid_walls):
        self.grid, self.grid_prev = step_diffusion_dufortfrankel(self.grid,self.grid_prev,grid_walls,self.diffuserate)
    
    #Decay the ligand using its halflife. Either return what has been decayed or not (slight overhead for returning)
    def decay(self,returndetail = False):
        if returndetail:
            self.grid, decaysites,decayvalues = step_decay_withreturn(self.grid,self.halflife)
            return decaysites,decayvalues
        else:
            self.grid = step_decay(self.grid,self.halflife)
            
    #Decay the ligand using a different grid, being careful not to go below 0
    #Could be sped up with numba if used intensively in the future
    def decaywithgrid(self,decaygrid,kd):
        self.grid = self.grid - kd*(self.grid * decaygrid)
        for i in range(1, self.grid.shape[0]-1):
            for j in range(1, self.grid.shape[1]-1):
                if self.grid[i,j]<0:
                    self.grid[i,j]=0
                #print(self.grid[i,j])
            
    #Remove the ligand with a specific df from the squares, either returning the values removed or not
    #Slight overhead for reporting back
    def consume(self,coveredsquares,consumptiondf,consumptionvmax,returndetail=False):
        if returndetail:
            self.grid, consumedvalues = step_consume_returnsites(self.grid,coveredsquares,consumptiondf,consumptionvmax)
            return consumedvalues
        else:
            self.grid, totalconsumed = step_consume(self.grid,coveredsquares,consumptiondf,consumptionvmax)
            return totalconsumed
    
    def consumelinear(self,coveredsquares,consumptiondf,returndetail=False):
        self.grid, totalconsumed = step_consume_linear(self.grid,coveredsquares,consumptiondf)
        return totalconsumed
    def consumeproportional(self,coveredsquares,consumptiondf,returndetail=False):
        self.grid, totalconsumed = step_consume_proportional(self.grid,coveredsquares)
        return totalconsumed
    #Sense the amount of this ligand in this area
    def sense(self,coveredsquares,kd=0):
        if kd:
            return sense_jitted_kd(self.grid,coveredsquares,kd)
        else:
            return sense_jitted(self.grid,coveredsquares)
        
    #Put a single amount of ligand into the system at the specified locations
    def produce_singlevalue(self,coveredsquares,production):
        productionvalues = np.full(len(coveredsquares),production/len(coveredsquares))
        self.grid = step_produce(self.grid,coveredsquares,productionvalues)
        
    #Put the specified amount of ligand into the system for specific locations
    def produce_multivalues(self,coveredsquares,productionvalues):
        self.grid = step_produce(self.grid,coveredsquares,productionvalues)


    