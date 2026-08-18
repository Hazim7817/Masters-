#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:39:00 2024
Change on 31 oct: add option for higher initial attractant concentration

@author: dmv
"""
#This is a simulation of macrophages crossing the bridge of an Insall chamber

#these are libraries from the virtual environment
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numba
import math
import argparse

#these are supporting files in the local folder
import mazelayouts
import environment
import cell
import collisionfunctions
import datasaver


#initialise our arguments
parser = argparse.ArgumentParser()
#General
parser.add_argument("-folder",default="fastparameters", help="Where to save the output")
parser.add_argument("-plotting",default = 25,help="How often to make a plot, 0 = never 1 = every timestep 2 = every two timesteps etc.",type=int)
parser.add_argument("-saving",default = 25,help="How often to save data",type=int)

#Environment   
parser.add_argument("-Nx",default=100, help="Width of the grid, default is 200",type=int)
parser.add_argument("-Ny",default=200, help="Height of the grid, default is 200",type=int)
parser.add_argument("-steps",default=1000, help="Duration of the simulation, default is 10000",type=int)
parser.add_argument("-cells",default=40, help="Maximum number of cells to place, default is 100",type=int)
parser.add_argument("-gradient",default=0, help="What initial gradient to use for the attractant",type=int)
parser.add_argument("-attractantplacementarea",default=3, help="What initial gradient to use for the attractant",type=int)
parser.add_argument("-metaboliteremovalarea",default=0, help="What initial gradient to use for the attractant",type=int)


#General cell properties
parser.add_argument("-cellsize",default=3, help="Size of cells in grid site equivalents",type=float)
parser.add_argument("-mitogenfactor",default=0, help="How much more to grow with attractant",type=float)
parser.add_argument("-basemitosis",default=0, help="How often to try to divide",type=int)
parser.add_argument("-basedeath",default=0, help="How often to die",type=float)

#Cell movement properties
parser.add_argument("-collision",default=1, help="Whether to have collision detection. 0 = only for walls, 1= distance-based for cells, 2=shape-based for cells (would allow for complex shapes)",type=int)
parser.add_argument("-celldistancefactor",default=2, help="How much distance to keep in cell sizes if collision is on",type=float)
parser.add_argument("-persistence",default=0.9, help="How much cells want to continue their course (0-1)",type=float)
parser.add_argument("-movedistance",default=0.2, help="How far should cells move per step",type=float)

#Chemotaxis cell properties

parser.add_argument("-cellfuzzingrelative",default=0, help="How much fuzzier kd values should be made relative to calculated kd (0-~1)",type=float)
parser.add_argument("-cellfuzzingabsolute",default=0, help="How much extra noise should be added to kd values(0-~1)",type=float)
parser.add_argument("-envirofuzzingabsolute",default=0.5, help="How much extra noise should be added to sensed attractant values per attractant(0-~1)",type=float)


parser.add_argument("-attractantweighing",default=0, help="How strongly to sense the attractant",type=float)
parser.add_argument("-initialattractant",default=1, help="How much attractant to place",type=float)

parser.add_argument("-attractantconsumptionkd",default=0.5, help="kd for consumption",type=float)
parser.add_argument("-attractantconsumptionvmax",default=1, help="vmax for consumption",type=float)

parser.add_argument("-attractantreceptorkd",default=0.5, help="What kd to use for sensing. If 0, default to directly reading the concentration",type=float)
parser.add_argument("-attractantdiffusion",default=6, help="Diffusion of attractant",type=float)



parser.add_argument("-metabolitediffusion",default=6, help="Diffusion of metabolite",type=float)

parser.add_argument("-metaboliteweighing",default=1, help="How much to sense metabolite",type=int)
parser.add_argument("-metabolitehalflife",default=150, help="How long the attractant should last",type=int)
parser.add_argument("-metaboliteproductionratio",default=250, help="How much secondary attractant to produce from the primary?",type=float)
parser.add_argument("-metaboliteconsumptionkd",default=0, help="How much of the metabolite to consume? Set to 0 to disable",type=float)
parser.add_argument("-metaboliteconsumptionvmax",default=0, help="How much of the metabolite to consume? Set to 0 to disable",type=float)
parser.add_argument("-metaboliteproductionfrequency",default= 50, help="How often to produce metabolite. 0 can always produce the metabolite. So it means how frequent to produce the metabolites in terms of steps. 1 means that the metabolite gonna be produced at every 2nd step etc...",type=int)
parser.add_argument("-metaboliteproductiontreshold",default=0, help="The minimum amount of attractant that must be present to make a secondary",type=float)
parser.add_argument("-metabolitereceptorkd",default=0.5, help="What kd to use for sensing. If 0, default to directly reading the concentration",type=float)
parser.add_argument("-metaboliterelayfactor",default=0, help="How much metabolite to produce based on consumed/sensed metabolite",type=float)

# good to see what happens if we change the metabolite halflife, metabolite production ratio, and metabolite consumption v max

# in a way we can add inhibitors???
#Load arguments
args = parser.parse_args()

#initialise our datastorage
mydatasaver = datasaver.datasaver(args.folder)
plotinterval = args.plotting
saveinterval = args.saving
# Environment parameters
Nx = args.Nx
Ny = args.Ny
steps = args.steps
maxcells = args.cells 
gradienttype = args.gradient
attractantplacementarea = args.attractantplacementarea
metaboliteremovalarea= args.metaboliteremovalarea

#Cell properties
cellsize = args.cellsize
mitogenfactor = args.mitogenfactor
basemitosis = args.basemitosis
basedeath = args.basedeath
attractantreceptorkd = args.attractantreceptorkd
metabolitereceptorkd = args.metabolitereceptorkd

#Cell movement properties
cellcollision = args.collision
celldistance = cellsize*args.celldistancefactor
persistence = args.persistence
movedistance= args.movedistance
cellfuzzingrelative = args.cellfuzzingrelative
cellfuzzingabsolute = args.cellfuzzingabsolute
envirofuzzingabsolute = args.envirofuzzingabsolute


#parameters for ligands
#attractantparameters - this represents a primary attractant
attractanthalflife = 0 #how long-lived this ligand is - a high number means it breaks down slowly. disabled at 0
initialattractant = args.initialattractant

attractantdiffusion = args.attractantdiffusion#.5 #How fast it diffuses - a high number means it diffuses a lot
attractantconsumptionkd = args.attractantconsumptionkd
attractantconsumptionvmax = args.attractantconsumptionvmax
attractantweighing = float(args.attractantweighing)#How strongly this is sensed. A negative number acts as a metabolite1

#metabolite1 - this represents a secondary attractant
metabolite1halflife = args.metabolitehalflife
metabolite1diffusion = args.metabolitediffusion

metabolite1consumptionkd = args.metaboliteconsumptionkd
metabolite1consumptionvmax = args.metaboliteconsumptionvmax
metabolite1weighing = float(args.metaboliteweighing)
metaboliteproductionratio = args.metaboliteproductionratio
producefrequency = args.metaboliteproductionfrequency # if I put at a higher number it might reflect pulsing
metaboliteproductiontreshold = args.metaboliteproductiontreshold
metaboliterelayfactor = args.metaboliterelayfactor


#Set number of diffusion steps
#attractantdiffusion = 1
diffusionrepeats=1
maxdiffuse = 0.1
if attractantdiffusion > metabolite1diffusion:
    if attractantdiffusion>maxdiffuse:
        diffusionrepeats = math.ceil(attractantdiffusion/maxdiffuse)
        attractantdiffusion= attractantdiffusion/diffusionrepeats
        metabolite1diffusion = metabolite1diffusion/diffusionrepeats
else:
    if metabolite1diffusion>maxdiffuse:
        diffusionrepeats = math.ceil(metabolite1diffusion/maxdiffuse)
        metabolite1diffusion= metabolite1diffusion/diffusionrepeats
        attractantdiffusion = attractantdiffusion/diffusionrepeats
        
        
# Create the grid walls
grid_walls = mazelayouts.makeemptydish(Nx, Ny)
alloccupiedsites = np.array([])
wallsites = np.array([])

#Add walls to the set of occupied sites
for i in range(0,Nx):
    for j in range(0,Ny):
        if grid_walls[i,j]>0:
            alloccupiedsites = np.union1d(alloccupiedsites,j*Nx+i)
            wallsites = np.union1d(wallsites,j*Nx+i)


#Create the chemical grids
ligands = {}
ligands["attractant"] = environment.ligand(Nx,Ny,attractanthalflife,attractantdiffusion,grid_walls,"attractant")
ligands["metab1"] = environment.ligand(Nx,Ny,metabolite1halflife,metabolite1diffusion,grid_walls,"metab1")


#Fill part of the grid with attractant
for i in range(1, ligands["attractant"].grid.shape[0]-1):
     for j in range(1, ligands["attractant"].grid.shape[1]-1):
         if grid_walls[i,j]<1:
             if gradienttype ==0:
                ligands["attractant"].grid_prev[i,j] = initialattractant#abs(math.sin(j/20))
             if gradienttype ==1:
                ligands["attractant"].grid_prev[i,j] = ((Ny-j)/Ny)*initialattractant
             if gradienttype ==2:
                if j<0.2*Ny:
                    ligands["attractant"].grid_prev[i,j] = 2*initialattractant
             if gradienttype ==3:
                 ligands["attractant"].grid_prev[i,j] = ((((Ny-j)/Ny)+0.5)/20)*initialattractant
        
#Do a first diffusion step on the grid, this is neccesary to do dufort diffusion later
for ligand in ligands.values():
    ligand.diffuse_euler_init(grid_walls)

#Place the cells
cells = list()

#pre-allocate space is we're saving their central locations only
if cellcollision ==1:
    cellxlocations = np.zeros(maxcells*10) #Make a guess for how much space we'll need
    cellylocations = np.zeros(maxcells*10)
      
for i in range(0,maxcells):
    unplaced = True
    attempt = 0
    while unplaced:
        #test if new cell would not overlap with any other cells or walls
        xlocation =random.randint(int(0.05*Nx)+cellsize, int(0.95*Nx)-cellsize)
        ylocation =random.randint(Ny-40,Ny-10)
        newcell = cell.cell(xlocation,ylocation,cellsize,cellcollision,cellfuzzingrelative,cellfuzzingabsolute,Nx,Ny,i)
        newcell.lastproduced = random.randint(-1*producefrequency,0)
        if cellcollision == 0:
            cells.append(newcell)
            newcell.definesurfacesquares()
            break
        
        if cellcollision == 1:
            if collisionfunctions.check_collision_distance(i,xlocation,ylocation,cellxlocations,cellylocations,celldistance):
                cellxlocations[i] = xlocation
                cellylocations[i] = ylocation
                cells.append(newcell)
                newcell.definesurfacesquares()
                break
            
        if cellcollision == 2:
            newoccupiedsites = newcell.definesurfacesquares()
            if len(np.intersect1d(alloccupiedsites,newoccupiedsites))==0: #This means we've placed it at an available stop
                alloccupiedsites = np.union1d(alloccupiedsites,newcell.definesurfacesquares())
                cells.append(newcell)
                break
            
        #This is only triggered if cellcollision was >0 and we failed to place the cell
        attempt +=1
        if attempt > 100:
            del newcell
            print("Unable to place cell!")
            break

#Increment the counter with one, so that any new cells that are created will get a new number            
maxcells = maxcells + 1

#run the loop where the first one is kinda like stochastic or at a fix rate and compare which one is more better 
# most of the secondary autoattract is short live that's why it quite hard to put a fix number on it because in-vitro it's quite hard to record them cause we dont know how much being produced and how much being destroyed
# let's say u have 1000 being made and 900 being destroyed than the the number of attractant is 100 or might be that the number of attractant being made is 100 but none is destroyed so that's quite hard
# if we are able to actually check the pulsing would be great, in a way there is no way for us to actually check that the pulsing works lmao 

##Run the main loop of our simulation
for t in range(steps):
    start = time.time()
    for i in range(1, ligands["attractant"].grid.shape[0]-1):
        for j in range(1, ligands["attractant"].grid.shape[1]-1):
            if j<(attractantplacementarea):
                ligands["attractant"].grid_prev[i,j] = initialattractant
                ligands["attractant"].grid[i,j] = initialattractant
                #print("placed")
            if j<(metaboliteremovalarea):
                #ligands["attractant"].grid_prev[i,j] = 0
                #ligands["attractant"].grid[i,j] = 0
                ligands["metab1"].grid_prev[i,j] = 0
                ligands["metab1"].grid[i,j] = 0
            if gradienttype == 1:
                if j>380:
                    ligands["attractant"].grid_prev[i,j] = 0
                    ligands["attractant"].grid[i,j] = 0
                    ligands["metab1"].grid_prev[i,j] = 0
                    ligands["metab1"].grid[i,j] = 0
    
    startdiffuse = time.time()
    #Diffuse ligands
    for ligand in ligands.values():
        diffusionrepeated = 1
        ligand.diffuse_euler(grid_walls)
        while diffusionrepeated <diffusionrepeats:
            ligand.diffuse_dufort(grid_walls)
            #ligand.diffuse_euler(grid_walls)
            diffusionrepeated+=1
    enddiffuse = time.time()
    #Degrade ligands
    if(attractanthalflife>0):
        ligands["attractant"].decay()
    if(metabolite1halflife>0):
        ligands["metab1"].decay()
    
    
    
    
            
    #Create fuzzed ligands if neccesary
    if envirofuzzingabsolute>0:
        fuzzedligands = []
        for ligand in ligands.values():
            fuzzedligands.append(ligand.grid+np.random.uniform(0,envirofuzzingabsolute,size=ligand.grid.shape))
    
    startmove = time.time()
    #shuffle the list so we adress them in a random order
    random.shuffle(cells)
    #Have cells sense chemoattractant, set their target accordingly, consume and produce ligands, and move
    for thiscell in cells:
        #sensing
        if envirofuzzingabsolute>0:
            thiscell.sense_multiple_attractants_fuzzedgrids(fuzzedligands,[attractantweighing,metabolite1weighing],persistence,[[0],[1]],[attractantreceptorkd,metabolitereceptorkd])
        else:
            thiscell.sense_multiple_attractants(ligands,[attractantweighing,metabolite1weighing],persistence,[[0],[1]],[attractantreceptorkd,metabolitereceptorkd])
  
        #consuming
        totalconsumed = 0
        if attractantconsumptionvmax >0:
            consumedpersite = ligands["attractant"].consume(thiscell.occupiedsites, attractantconsumptionkd,attractantconsumptionvmax,returndetail=True)
            totalsensed = sum(consumedpersite)
            
        else:
            totalsensed = ligands["attractant"].sense(thiscell.occupiedsites,attractantreceptorkd)
            #print(totalsensed/len(thiscell.occupiedsites))
        if metabolite1consumptionvmax >0:
            metabconsumedpersite = ligands["metab1"].consume(thiscell.occupiedsites, metabolite1consumptionkd,metabolite1consumptionvmax,returndetail=True)
            totalsensed += metaboliterelayfactor * sum(metabconsumedpersite)
            thiscell.consumedthisstep = sum(metabconsumedpersite)
        else:
            metabsensed = ligands["metab1"].sense(thiscell.occupiedsites,metabolitereceptorkd)
            totalsensed += metaboliterelayfactor * metabsensed
        #producing
        if (t-thiscell.lastproduced)>producefrequency:
            #print(totalsensed)
            if (totalsensed/len(thiscell.occupiedsites))>metaboliteproductiontreshold:
                thiscell.lastproduced=t
                amount = totalsensed*metaboliteproductionratio
                thiscell.producedthisstep = amount
                thiscell.accumulatedproduction += amount
                ligands["metab1"].produce_singlevalue(thiscell.occupiedsites, amount)
            # I need to reset the pulsing after each pulses since this one just causes the cell to accumulate, maybe saves more data or resets it periodically (by making sure that i didn't miss any)
           
        #collision checking
        if thiscell.collision ==0:
            thiscell.move_simple(movedistance,grid_walls)
        if thiscell.collision ==1:
            newxloc,newyloc = thiscell.move_coarse(movedistance,Nx,Ny,grid_walls,cellxlocations,cellylocations,celldistance)
            if ((newxloc == cellxlocations[thiscell.id]) and (newyloc == cellylocations[thiscell.id])):
                thiscell.targetangle = random.uniform(-1*math.pi,math.pi)
            else:
                cellxlocations[thiscell.id], cellylocations[thiscell.id] = newxloc,newyloc
             
        if thiscell.collision ==2:
            alloccupiedsites = thiscell.move_fine(movedistance,Nx,Ny,alloccupiedsites)
            
        #Consider mitosis based on local attractant consumed
        if random.random() <(basemitosis+mitogenfactor*totalconsumed):
                if thiscell.collision == 0:
                    cells,maxcells = thiscell.divide_simple(cells,maxcells)
                if thiscell.collision == 1:
                    cells,cellxlocations,cellylocations,maxcells = thiscell.divide_coarse(cells,maxcells,cellxlocations,cellylocations,celldistance*2)
                if thiscell.collision == 2:
                    cells, alloccupiedsites,maxcells = thiscell.divide_fine(alloccupiedsites,cells,maxcells)
        
        #consider dying
        if random.random()<basedeath:
            if thiscell.collision == 1:
                cellxlocations[thiscell.id] = -1
                cellylocations[thiscell.id] = -1
            if thiscell.collision == 2:
                alloccupiedsites = np.setxor1d(alloccupiedsites,thiscell.occupiedsites)
            thiscell.alive = False
        
    #Take out our dead cells
    cells[:] = [thiscell for thiscell in cells if thiscell.alive]
    
  
    
    #check if we need to add more space to the cellocationarray
    if cellcollision == 1:
        if (maxcells*2) > (len(cellxlocations)):
            cellxlocations = np.append(cellxlocations,np.zeros(len(cellxlocations)))
            cellylocations = np.append(cellylocations,np.zeros(len(cellxlocations)))
    
    #timekeeping
    endmove = time.time()
    end = time.time()
    
    #Save data every this many steps using feather
    if(t%saveinterval==0):
        attractantname = "attractant"
        metabolitename = "metabolite"
        cellname = "cells"
        startsave = time.time()
        mydatasaver.savegrid(ligands["attractant"].grid, attractantname+ str(t))
        mydatasaver.savegrid(ligands["metab1"].grid, metabolitename+ str(t))
        mydatasaver.savecellsextended(cells, cellname+ str(t))
        for thiscell in cells:
            thiscell.producedthisstep = 0
            thiscell.accumulatedproduction = 0
        
        #MAYBE RESET IT AFTER
        # ADd the production value before and after  
        endsave = time.time()
    
    #mydatasaver.savecellscsv(cells,"tracking",t)
    #Plot the current state every so many steps
    if plotinterval >0:
        if(t%plotinterval==0):
            print("step")
            print(t)
            print("totaltime")
            print(end - start)
            print("diffusetime")
            print(enddiffuse-startdiffuse)
            print("movetime")
            print(endmove-startmove)
            print("savetime (not included in total)")
            print(endsave-startsave)
            #print("Totalattractant")
            #print(sum(sum(ligands["attractant"].grid.T)))
            #print("totalmetab")
            #print(sum(sum(ligands["metab1"].grid.T)))
            #print("totalstuff")
            #print(sum(sum(ligands["metab1"].grid.T))+sum(sum(ligands["attractant"].grid.T)))
            startplot= time.time()
            plt.clf()
            plt.imshow(ligands["attractant"].grid.T+ligands["metab1"].grid.T, origin='lower', extent=[0, ligands["attractant"].grid.shape[0], 0,ligands["attractant"].grid.shape[1]], cmap='viridis', interpolation='none',vmin=0,vmax=1)
            #plt.imshow(ligands["metab1"].grid.T, origin='lower', extent=[0, ligands["attractant"].grid.shape[0], 0,ligands["attractant"].grid.shape[1]], cmap='viridis', interpolation='none',vmin=0,vmax=1)
            #plt.imshow(ligands["attractant"].grid.T, origin='lower', extent=[0, ligands["attractant"].grid.shape[0], 0,ligands["attractant"].grid.shape[1]], cmap='viridis', interpolation='none',vmin=0,vmax=1)
            
            #plt.colorbar(label="Attractiveness")
            plt.title("Time = "+str(t)+" steps")
            for thiscell in cells:
                cellcircle = patches.Circle((thiscell.xlocation, thiscell.ylocation), thiscell.size, linewidth=1, edgecolor='g', facecolor='none')
                plt.gca().add_patch(cellcircle)
            plt.show()
            endplot = time.time()
            print("plottime (not included)")
            print(endplot-startplot)
            #plt.pause(0.000001)



# Final plot after the loop
plt.show()
