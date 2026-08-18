#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:40:58 2024

@author: dmv
"""
import numpy as np
import numba
import random
import math
import collisionfunctions
from numba.typed import List

#This creates a target angle based on a target destination
#determined using the weighted concentrations of attractants in the environment
@numba.jit(nopython=True)
def sense_multiple_attractants_average_fuzzy_jitted(occupiedsites,xlocation,ylocation,size,ligands,weighing,relativefuzzing,absolutefuzzing,receptors,receptorkds):
    totalsquares = 0
    totalxvalues = 0
    totalyvalues = 0
    totalxlocations =0
    totalylocations = 0
    totalattractant = 0
    Nx = ligands[0].shape[0]
    for point in occupiedsites:
        i = point%Nx
        j = math.floor(point / Nx)
        #gridnumber = 0
        netattraction = 0.0
        totalsquares +=1
        totalxlocations +=i
        totalylocations +=j
        for thisreceptor in receptors:
            if len(thisreceptor)==1:
                thisligand = ligands[thisreceptor[0]]
                if receptorkds[thisreceptor[0]]>0:
                    perceivedattractant = (thisligand[i,j]/(thisligand[i,j]+receptorkds[thisreceptor[0]]))*weighing[thisreceptor[0]]
                else:
                    perceivedattractant = (thisligand[i,j])*weighing[thisreceptor[0]]
            else:
                totalpotentialefficacy = 0
                totalpotentialbound = 0
                for ligandnumber in thisreceptor:
                    thisligand = ligands[ligandnumber]
                    totalpotentialefficacy += ((weighing[ligandnumber]*thisligand[i,j])/receptorkds[ligandnumber])
                    totalpotentialbound += (thisligand[i,j]/receptorkds[ligandnumber])
                perceivedattractant = (totalpotentialefficacy/(totalpotentialbound+1))
  
            netattraction += perceivedattractant

        netattraction *= random.uniform(1-relativefuzzing,1+relativefuzzing)
        netattraction += random.uniform(0,absolutefuzzing)
  

        #print(netattraction)
        if netattraction <0:
            netattraction=0 #Negative attraction values lead to a lot of weird effects
        
        totalattractant += netattraction
        totalxvalues += netattraction*i
        totalyvalues += netattraction*j
        
    if totalxvalues !=0 or totalyvalues != 0:
        desiredx = totalxvalues/totalattractant
        desiredy = totalyvalues/totalattractant
        xlocationformovement = totalxlocations/totalsquares
        ylocationformovement = totalylocations/totalsquares
        xdirection = desiredx-xlocationformovement
        ydirection = desiredy-ylocationformovement
        targetangle = math.atan2(ydirection, xdirection)
        return targetangle
    print("Did not find angle")
    return random.uniform(-1*math.pi,math.pi)

#This returns the squares that are covered by this cell, used for determining where to consume and collide and such
@numba.jit(nopython=True)
def definesurfacesquares_jitted(xlocation,ylocation,size,Nx,Ny):
    occupiedsites = np.zeros(math.ceil(size)*math.ceil(size)*5,dtype=np.int64)
    sitecount = 0
    for i in range(max(math.floor(xlocation-size),0), min(math.ceil(xlocation+size),Nx-1)):
        for j in range(max(math.floor(ylocation-size),0),min(math.ceil(ylocation+size),Ny-1)):
            if math.sqrt(pow(xlocation-i,2)+pow(ylocation-j,2))<size:
                occupiedsites[sitecount]= j*Nx+i
                sitecount +=1
    return(occupiedsites[0:sitecount])


#Calculate a new angle between previous angles using a persistence value
@numba.jit(nopython=True)
def make_angle_jitted(targetangle,newangle,persistence):
    x = y = 0
    x += math.cos(targetangle)*persistence
    y += math.sin(targetangle)*persistence
    
    x += math.cos(newangle)*(1-persistence)
    y += math.sin(newangle)*(1-persistence)

    return math.atan2(y, x)

#This is used for a simple move that does not need to be checked for a collision, which is much faster
@numba.jit(nopython=True)    
def move_jitted(maxdistance,xlimit,ylimit,targetangle,xlocation,ylocation,size):
    delta_x = maxdistance * math.cos(targetangle)
    delta_y = maxdistance * math.sin(targetangle)
    targetx = xlocation+delta_x
    targety = ylocation+delta_y
    #If any of these are true, the move is out of bounds and should not happen
    if (targetx >=(xlimit-size)) or (targetx <=size):
        return xlocation,ylocation
    if (targety >=(ylimit-size)) or (targety <=size):
        return xlocation,ylocation
    return targetx,targety

#This is used for a move that can be checked for collision simply by distance, which is fast
@numba.jit(nopython=True)    
def move_coarse_collision_detection_jitted(myid,maxdistance,xlimit,ylimit,targetangle,xlocation,ylocation,xlocs,ylocs,maxproximity):
    delta_x = maxdistance * math.cos(targetangle)
    delta_y = maxdistance * math.sin(targetangle)
    targetx = xlocation+delta_x
    targety = ylocation+delta_y
    if not collisionfunctions.check_collision_distance(myid,targetx,targety,xlocs,ylocs,maxproximity):
        return xlocation,ylocation
        
    #If any of these are true, the move is out of bounds and should not happen
    if (targetx >=(xlimit-maxproximity)) or (targetx <maxproximity):
        return xlocation,ylocation
    if (targety >=(ylimit-maxproximity)) or (targety <maxproximity):
        return xlocation,ylocation
    return targetx,targety

#This is used for a move that does need to be checked for a collision, which is much slower
#This does allow for much more complex cell shapes
#Currently not jittable, might be done in the future (but may not be much more efficient)
def move_with_fine_collision_detection(maxdistance,xlimit,ylimit,targetangle,size,xlocation,ylocation,selfoccupiedsites,occupiedsites):
    delta_x = maxdistance * math.cos(targetangle)
    delta_y = maxdistance * math.sin(targetangle)
    targetx = xlocation+delta_x
    targety = ylocation+delta_y
    #If any of these are true, the move is out of bounds and should not happen
    if (targetx >= xlimit) or (targetx <0):
        return occupiedsites,selfoccupiedsites,xlocation,ylocation
    if (targety >= ylimit) or (targety <0):
        return occupiedsites, selfoccupiedsites,xlocation,ylocation
    
    #check what new sites we would occupy
    newoccupiedsites = definesurfacesquares_jitted(targetx,targety,size,xlimit,ylimit) 
    movepossible = False
    #use this collisionfunction to check if that overlaps with already occupied sites that aren't our own
    movepossible, occupiedsites = collisionfunctions.checkoverlapcollision(occupiedsites,selfoccupiedsites,newoccupiedsites)
    if movepossible: 
        selfoccupiedsites = newoccupiedsites #set new location
        xlocation=targetx
        ylocation=targety
    return occupiedsites, selfoccupiedsites,xlocation,ylocation

#TODO
#def divide_simple_jitted()
#Currently not jitted, but would be faster if it was, probably

class cell:
    def __init__(self,xlocation,ylocation,size,collision,relativefuzzing,absolutefuzzing,Nx,Ny,idnumber=0):
        self.xlocation = xlocation #This can be a floating point number
        self.ylocation = ylocation #This can be a floating point number
        self.Nx = Nx#How big is my environment?
        self.Ny = Ny
        self.size = size #This is a radius
        self.targetangle = random.uniform(-1*math.pi,math.pi)
        self.occupiedsites  = list()#This has to be kept updated through the below method
        self.collision = collision
        self.relativefuzzing = relativefuzzing
        self.absolutefuzzing = absolutefuzzing
        self.distance = 0 #distance moved in the last timestep
        self.id=idnumber #a unique number
        self.alive = True
        self.lastproduced = random.randint(-1000, -10) #what timestep this cell has last produced chemoattractant at
        self.receptorpresence = 1 #Between 1 and 0, represents how many receptors remain
        self.consumedthisstep = 0 #how much secondary the cell has produced this step
        self.producedthisstep = 0 #how much secondary the cell has produced this step
        self.accumulatedproduction = 0 #how much secondary the cell has produced 
    #Return the current squares occupied by this cell
    def definesurfacesquares(self): #Wrapper to allow for JIT enhancement of underlying function
        self.occupiedsites = definesurfacesquares_jitted(self.xlocation,self.ylocation,self.size,self.Nx,self.Ny)
        return self.occupiedsites
    
    #Return the angle corresponding to the local gradient of multiple attractants combined
    def sense_multiple_attractants(self,grids,weighing,persistence,receptors,receptorkd=0,):#Wrapper to allow for JIT enhancement of underlying function
        #Sense the gradient and get a new angle from that
        receptors = List(List(x) for x in receptors) 
        newangle = sense_multiple_attractants_average_fuzzy_jitted(self.occupiedsites,self.xlocation,self.ylocation,self.size,[ligand.grid for ligand in grids.values()],weighing,self.relativefuzzing,self.absolutefuzzing,receptors,receptorkd)
        #Set a new target angle based on our previous angle, the new angle, and how persistent we are
        self.targetangle = make_angle_jitted(self.targetangle,newangle,persistence)
    
    #Return the angle corresponding to the local gradient of multiple attractants combined
    def sense_multiple_attractants_fuzzedgrids(self,grids,weighing,persistence,receptors,receptorkd=0,):#Wrapper to allow for JIT enhancement of underlying function
        #Sense the gradient and get a new angle from that
        receptors = List(List(x) for x in receptors) 
        newangle = sense_multiple_attractants_average_fuzzy_jitted(self.occupiedsites,self.xlocation,self.ylocation,self.size,grids,weighing,self.relativefuzzing,self.absolutefuzzing,receptors,receptorkd)
        #Set a new target angle based on our previous angle, the new angle, and how persistent we are
        self.targetangle = make_angle_jitted(self.targetangle,newangle,persistence)
            
    #Move without complex collision detection, but avoiding moving into walls
    def move_simple(self,maxdistance,walls):
        targetx,targety = move_jitted(maxdistance,self.Nx,self.Ny,self.targetangle,self.xlocation,self.ylocation,self.size)
        newoccupiedsites = definesurfacesquares_jitted(targetx,targety,self.size,self.Nx,self.Ny)
        
        if collisionfunctions.wall_collision_test_jitted(newoccupiedsites,walls,self.Nx)==False:
            self.xlocation, self.ylocation = targetx,targety
            self.occupiedsites = newoccupiedsites
    
    #Move with a simple center-based collision detection
    def move_coarse(self,maxdistance,xlimit,ylimit,walls,xlocs,ylocs,maxproximity):
        targetx,targety = move_coarse_collision_detection_jitted(self.id,maxdistance,xlimit,ylimit,self.targetangle,self.xlocation,self.ylocation,xlocs,ylocs,maxproximity)
        #print(targetx,targety)
        newoccupiedsites = definesurfacesquares_jitted(targetx,targety,self.size,self.Nx,self.Ny)
        if collisionfunctions.wall_collision_test_jitted(newoccupiedsites,walls,self.Nx)==False:
            self.xlocation, self.ylocation = targetx,targety
            self.occupiedsites = newoccupiedsites
        return targetx, targety
    
    #Move with precise shape-based collision detection
    def move_fine(self,maxdistance,xlimit,ylimit,occupiedsites): 
        occupiedsites, self.occupiedsites, self.xlocation, self.ylocation = move_with_fine_collision_detection(maxdistance,xlimit,ylimit,self.targetangle,self.size,self.xlocation, self.ylocation,self.occupiedsites,occupiedsites)
        return occupiedsites

    #Divide without collision detection (can divide into walls, but not outside of area)
    def divide_simple(self,cells,maxcells):
       dividetries = 0
       while dividetries <2:
           dividetries +=1
           #Pick a random angle
           mitosisangle = random.uniform(-1*math.pi,math.pi)
           delta_x = self.size*2 * math.cos(mitosisangle)
           delta_y = self.size*2 * math.sin(mitosisangle)
           xlocation = self.xlocation + delta_x
           ylocation = self.ylocation + delta_y
           if xlocation>self.Nx or ylocation>self.Ny:
               continue
           if xlocation<0 or ylocation<0:
               continue
           newcell = cell(xlocation,ylocation,self.size,self.collision,self.fuzzing,self.Nx,self.Ny,maxcells)
           cells.append(newcell)
           newcell.definesurfacesquares()
           maxcells = maxcells + 1
           return cells,maxcells
    
    #Divide with coarse collision detection
    def divide_coarse(self,cells,maxcells,cellxlocations,cellylocations,celldistance):
        dividetries = 0
        while dividetries <2:
            dividetries +=1
            #Pick a random angle
            mitosisangle = random.uniform(-1*math.pi,math.pi)
            delta_x = max(self.size*2,celldistance) * math.cos(mitosisangle)
            delta_y = max(self.size*2,celldistance) * math.sin(mitosisangle)
            xlocation = self.xlocation + delta_x
            ylocation = self.ylocation + delta_y
            if xlocation>self.Nx or ylocation>self.Ny:
                continue
            if xlocation<0 or ylocation<0:
                continue
            if collisionfunctions.check_collision_distance(-1,xlocation,ylocation,cellxlocations,cellylocations,celldistance):
                newcell = cell(xlocation,ylocation,self.size,self.collision,self.fuzzing,self.Nx,self.Ny,maxcells)
                cellxlocations[maxcells] = xlocation
                cellylocations[maxcells] = ylocation
                cells.append(newcell)
                newcell.definesurfacesquares()
                maxcells = maxcells + 1
                break
        return cells, cellxlocations,cellylocations,maxcells
    
    #Divide with fine collision detection
    def divide_fine(self,alloccupiedsites,cells,maxcells):
        dividetries = 0
        while dividetries <2:
            dividetries +=1
            #Pick a random angle
            mitosisangle = random.uniform(-1*math.pi,math.pi)
            delta_x = self.size*2 * math.cos(mitosisangle)
            delta_y = self.size*2 * math.sin(mitosisangle)
            xlocation = self.xlocation + delta_x
            ylocation = self.ylocation + delta_y
            if xlocation>self.Nx or ylocation>self.Ny:
                continue
            if xlocation<0 or ylocation<0:
                continue
            newcell = cell.cell(xlocation,ylocation,self.cellsize,self.cellcollision,self.cellfuzzing,self.Nx,self.Ny,maxcells)
            newoccupiedsites = newcell.definesurfacesquares()
            if sum(np.isin(alloccupiedsites,newoccupiedsites))==0:
                alloccupiedsites = np.union1d(alloccupiedsites,newoccupiedsites)
                cells.append(newcell)
                maxcells = maxcells + 1 #Increment for the next cell only if previous number is actually used
                break
        return cells,alloccupiedsites,maxcells
    
    #remove self
    def die(self):
        del self
       
        