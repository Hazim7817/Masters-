# Autoattractant Migration Simulation

Simulation code accompanying: *Investigating the effects of autoattractant release timing and magnitude on collective immune-cell chemotaxis.*

This is an agent-based model of immune cell chemotaxis in which cells migrate along self-generated gradients of a primary attractant and can also produce, sense, and consume a secondary autoattractant. The code simulates a rectangular chemotaxis chamber and tracks cell positions and attractant concentrations over time.

## Requirements

Python 3 with the following packages:

- `numpy`
- `numba`
- `matplotlib`
- `pyarrow` (for saving output in Feather format)
- `pandas`

Install these into a virtual environment:

python -m venv .venv
source .venv/bin/activate
pip install numpy numba matplotlib pyarrow pandas

Whenever running a simulation, initialize the virtual environment first by again running

source .venv/bin/activate

## Files

All files should be in the same directory.

- `simulation_migration.py` — Main simulation script, run from the command line.
- `cell.py` — Cell class and movement/sensing logic (Numba-accelerated).
- `environment.py` — Ligand class handling diffusion, decay, consumption, and production on the lattice (Numba-accelerated).
- `collisionfunctions.py` — Collision detection between cells and walls.
- `mazelayouts.py` — Functions that generate wall layouts for the simulation domain.
- `datasaver.py` — Saves concentration grids and cell positions to Apache Feather files for analysis.


## Running a single simulation

The simulation is controlled entirely through command-line arguments. A minimal example:

python simulation_migration.py   -folder "my_output_folder" -Nx 100 -Ny 200 -cells 40 -steps 1000   -attractantdiffusion 6 -metabolitediffusion 6   -metabolitehalflife 50 -metaboliteproductionratio 30  -plotting 0 -saving 25


This runs a 1000-step simulation on a 100×200 grid with 40 cells, saves snapshots every 25 steps to `./my_output_folder/`, and produces no plots during the run.

## Command-line parameters

### General

| Flag | Default | Description |
|---|---|---|
| `-folder` | `testbed` | Output directory (created automatically) |
| `-plotting` | `100` | Plot every N steps. Set to `0` to disable plotting entirely (recommended for batch runs). |
| `-saving` | `500` | Save concentration grids and cell data every N steps. |

### Environment

| Flag | Default | Description |
|---|---|---|
| `-Nx` | `100` | Grid width in lattice sites. Each site represents 10 µm. | Not changed from default value in paper
| `-Ny` | `200` | Grid height in lattice sites. | Not changed from default value in paper
| `-steps` | `20000` | Number of simulation timesteps. Each timestep represents 6 s. |
| `-cells` | `40` | Number of cells to place at the start. |
| `-gradient` | `0` | Initial attractant distribution. `0` = uniform, `1` = linear gradient. | Not used in paper
| `-attractantplacementarea` | `3` | Number of rows at the bottom where attractant is continuously replenished (acts as a source). | Not changed from default value in paper
| `-metaboliteremovalarea` | `0` | Number of rows at the bottom where autoattractant is removed each step (acts as a sink). Set to `0` to disable. | Not used in paper
| `-initialattractant` | `1` | Initial concentration of primary attractant. |

### Cell properties

| Flag | Default | Description |
|---|---|---|
| `-cellsize` | `3` | Cell radius in lattice sites. At 10 µm per site this gives a 50 µm sensing diameter, as in the paper. | Not changed from default value in paper
| `-movedistance` | `0.01` | Distance moved per timestep in lattice units. `0.2` gives ~20 µm/min (fast cells); `0.01` gives ~1 µm/min (slow cells). |
| `-persistence` | `0.9` | Directional persistence (0–1). The new heading is a weighted average of the previous heading and the sensed gradient direction. | Not changed from default value in paper
| `-collision` | `1` | Collision mode. `0` = walls only, `1` = distance-based cell–cell (used in the paper), `2` = shape-based. | Not changed from default value in paper
| `-celldistancefactor` | `2` | Minimum centre-to-centre distance between cells, in multiples of cell radius. | Not changed from default value in paper
| `-basemitosis` | `0` | Base probability of cell division per step. `0` disables division. | Not used in paper
| `-basedeath` | `0` | Probability of cell death per step. `0` disables death. | Not used in paper
| `-mitogenfactor` | `0` | Additional division probability scaling with consumed attractant. | Not used in paper

### Sensing and noise

| Flag | Default | Description |
|---|---|---|
| `-envirofuzzingabsolute` | `0.5` | Absolute noise added to the attractant grid before sensing. This prevents cells from following arbitrarily shallow gradients. |  Not changed from default value in paper
| `-cellfuzzingrelative` | `0` | Relative noise applied to each cell's perceived attractant (multiplicative). | Not used in paper
| `-cellfuzzingabsolute` | `0` | Absolute noise applied to each cell's perceived attractant (additive). | Not used in paper

### Primary attractant

| Flag | Default | Description |
|---|---|---|
| `-attractantdiffusion` | `30` | Diffusion coefficient of primary attractant (in simulation units; subdivided internally for numerical stability). |
| `-attractantconsumptionkd` | `0.5` | Michaelis constant (Km) for consumption of primary attractant. |  Not changed from default value in paper
| `-attractantconsumptionvmax` | `0.05` | Vmax for consumption of primary attractant. `1` for fast cells, `0.05` for slow cells. |
| `-attractantreceptorkd` | `0.5` | Receptor Kd for sensing primary attractant. |  Not changed from default value in paper
| `-attractantweighing` | `1` | Relative weight of the primary attractant in direction sensing. |  Not changed from default value in paper

### Autoattractant (referred to as "metabolite" in the code)

| Flag | Default | Description |
|---|---|---|
| `-metabolitediffusion` | `6` | Diffusion coefficient of the autoattractant. |
| `-metabolitehalflife` | `75` | Half-life of autoattractant in timesteps. Controls decay rate. Set to `0` to disable decay (autoattractant persists unless consumed). |
| `-metaboliteproductionratio` | `50` | How much autoattractant is produced per unit of primary attractant consumed. This is the parameter α in the paper. |
| `-metaboliteconsumptionkd` | `0.5` | Km for cellular consumption of autoattractant. |  Not changed from default value in paper
| `-metaboliteconsumptionvmax` | `0` | Vmax for consumption of autoattractant. Set to `0` to disable consumption (autoattractant only decays). |
| `-metaboliteweighing` | `1` | Relative weight of autoattractant in direction sensing. |  Not changed from default value in paper
| `-metabolitereceptorkd` | `0.5` | Receptor Kd for sensing autoattractant. |  Not changed from default value in paper
| `-metaboliteproductionfrequency` | `0` | Minimum number of steps between production events per cell. `0` means cells produce every step. | Not used in paper
| `-metaboliteproductiontreshold` | `0` | Minimum local attractant concentration (per occupied site) required for a cell to produce autoattractant. | Not used in paper
| `-metaboliterelayfactor` | `0` | Amount of autoattractant produced based on consumed/sensed autoattractant (relay production). `0` disables relay. | Not used in paper

## Reproducing the paper's parameter sets

The paper uses two cell types. The key differences from the defaults are:

**Fast-moving cells (e.g. neutrophils), 1000 steps:**

python simulation_migration.py  -Nx 400 -Ny 200 -cells 40 -steps 1000 -movedistance 0.2 -persistence 0.9 -attractantconsumptionvmax 1 -attractantconsumptionkd 0.5 -attractantreceptorkd 0.5 -attractantdiffusion 6 -envirofuzzingabsolute 0.5


**Slow-moving cells (e.g. macrophages), 20000 steps:**

python simulation_migration.py  -Nx 400 -Ny 200 -cells 40 -steps 20000 -movedistance 0.01 -persistence 0.9 -attractantconsumptionvmax 0.05 -attractantconsumptionkd 0.5 -attractantreceptorkd 0.5 -attractantdiffusion 6 -envirofuzzingabsolute 0.5


The autoattractant parameters (`-metaboliteproductionratio`, `-metabolitehalflife`, `-metaboliteconsumptionvmax`, `-metabolitediffusion`) are then varied across simulations. Setting `-metaboliteproductionratio 0` runs a control simulation without autoattractant.

## Running parameter sweeps

The paper's results are generated from large parameter sweeps. Each condition is repeated 10 times with different random seeds. The supplied shell script `run_simulation.sh` shows how to do this in practice. It loops over production ratios and half-lives, launching 10 replicates in parallel for each combination. Each inner loop of 10 replicates runs in parallel (note the `&`), and `wait` ensures all jobs for one production/decay combination finish before moving on. Adjust the level of parallelism to match your available cores.

## Output

Each simulation creates its output directory (specified by `-folder`) containing Feather-format files saved at the interval set by `-saving`:

- `attractantN` — Flattened primary attractant concentration grid at timestep N.
- `metaboliteN` — Flattened autoattractant concentration grid at timestep N.
- `cellsN` — Cell positions (x, y), IDs, and per-step production/consumption values at timestep N.

Grids are stored as flattened 1D arrays of length Nx × Ny and should be reshaped to (Nx, Ny) for analysis.

## Mapping simulation units to physical units

The simulation operates on a dimensionless lattice. To convert to the physical units used in the paper:

- 1 lattice site = 10 µm (so a 100 × 200 grid = 1000 × 2000 µm)
- 1 timestep = 6 s
- Concentrations are in normalised units; 1 unit = 20 nM in the paper's standard conditions

## Notes

- The first timestep includes Numba JIT compilation, which adds a one-off delay of a few seconds.
- Setting `-plotting 0` is neccesary for batch runs, as matplotlib rendering pauses the simulation unless you're running an IDE that captures the images (such as Spyder)
