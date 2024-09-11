# preprocessing
SPACING = 1.0, 1.0, 2.0
PATCH_SIZE = 128, 128, 32
# PATCH_SIZE = 32,32,32
WINDOW_HU = -1350, 1000

# pre-training
MIN_WINDOW_HU = -1000, 300
MAX_WINDOW_HU = -1350, 1000

# MIN_WINDOW_HU = -2000, 300
# MAX_WINDOW_HU = -2000, 1000
# 
MAX_NUM_VOXELS_PER_PATCH = 1
# MAX_NUM_VOXELS_PER_PATCH = 27

# architecture
BASE_CHANNELS = 16
# BASE_CHANNELS = 64
NUM_SCALES = 6
# NUM_SCALES =1
