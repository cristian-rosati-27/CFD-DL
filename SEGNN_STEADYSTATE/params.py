
import os
import torch

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_files(directory):
    num_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    return num_files

### CONSTANTS

NSTEP = 10500
NSTEP = 7000
SPACING = 0.0008
NFRAME = 6 # depends on the saving frequency of probes, usually 1

NPOS = 6  # pos, mis, sdf, size (2 if fluid_node, 1 if outlet_node, 0 if inlet_node)
NFEATURES = 10  # p, vel, stress # TOGLI COLONNE NULLE

# introduzione e stato dell'arte in ambito aerospaziale (motivazione!!)
# prima pagina abstract introduzione e una colonnad istato dell'arte
###

# NPOS SHOULD BE 6 VALUES (npy file) -> x, y, z, mis, sdf, size (2 if fluid_node, 1 if outlet_node, 0 if inlet_node)
# NFEATURES SHOULD BE 10 VALUES (npz file with each feature as an array for all the nodes of that sim) -> p, vel_x, vel_y, vel_z, stress_xx, stress_xy, stress_yy, stress_xz, stress_yz, stress_zz

### HYPERPARAMETERS

EPOCHS = 10
BATCH_SIZE = 1
INPUT_SIZE = 11 # <- only for reference
EDGE_LMAX = 1
NODE_LMAX = 1
HIDDEN_LMAX = 1
NUM_LAYERS = 4
TASK = 'node'
NORM = 'batch'
OUTPUT_SIZE = 4
HIDDEN_SIZE= 64
NEIGHBOURS = 10 # 10 for moebius data
SUBSAMPLE_DATASET = 1
OPT = 'Adam'
SCHEDULER = 'ReduceLROnPlateau' # 'ExponentialLR' # None 
LEARNING_RATE= 3e-3 # 5e-5
if SCHEDULER == 'ExponentialLR':
    GAMMA = 0.95 # 0.8317 25 epochs, 0.8913 40 epochs 0.9261 60 epoch
if SCHEDULER == 'ReduceLROnPlateau':
    PATIENCE = 2
    FACTOR = 0.1 # new_lr = lr * factor
DEVICE = dev
EARLY_STOP = EPOCHS

## SEGNN_MODEL

MODEL_DIR = os.path.join("../.data/.SEGNN_MODEL")

### DATASET

DATASET = 3

if DATASET == 1:
    NFLUID_PROBES = 848
    NIO_PROBES = 0
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    DATADIR = os.path.join("../.data/Extracted_data")
    NSIM = 40

if DATASET == 2:
    NFLUID_PROBES = 848
    NIO_PROBES = 0
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    DATADIR = os.path.join("../.data/Dataset_50sims")
    NSIM = 50

if DATASET == 3:
    NFLUID_PROBES = 848
    NIO_PROBES = 0
    INLET_ID = 0  # inlet id
    NFRAME = 1 

    DATADIR = os.path.join("../.data/Dataset_75sims")
    NSIM = 75