import os

# Configurations for inference
INFERENCE_WEIGHTS_PATH = ''               # Path to weights for inference# Folder to save output files
NUM_SAMPLES = 1000                                               # Number of samples to generate (only for generate mode)
TOP_K = 9                                                       # Top k for sampling
TOP_P = 0.9                                                      # Top p for sampling
TEMPERATURE = 1.2                                                 # Temperature for sampling
ORIGINAL_OUTPUT_FOLDER = os.path.join('../output/original', os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))
INTERLEAVED_OUTPUT_FOLDER = os.path.join('../output/interleaved', os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))

# Configurations for model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                               # Hidden Size