import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['bmicdl05']

# ==================================================================
# project dirs
# ==================================================================
project_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/'
bmic_data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/'
project_code_root = os.path.join(project_root, 'methods/tto_ss/tmp/')
project_data_root = os.path.join(project_root, 'data/')

# ==================================================================
# data dirs
# ==================================================================
orig_data_root_nci = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate'
orig_data_root_pirad_erc = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/Prostate/'
orig_data_root_promise = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/Prostate_PROMISE12/TrainingData/'

# ==================================================================
# dirs where the pre-processed data is stored
# ==================================================================
preproc_folder_nci = os.path.join(project_data_root,'preprocessed/nci/')
preproc_folder_pirad_erc = os.path.join(project_data_root,'preprocessed/pirad_erc/')
preproc_folder_promise = os.path.join(project_data_root,'preprocessed/promise/')