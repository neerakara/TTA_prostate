import model_zoo
import tensorflow as tf

# ======================================================================
# test settings
# ======================================================================
train_dataset = 'NCI'
test_dataset = 'PIRAD_ERC'
whole_gland_results = True
normalize = True
run_number = 1

# ====================================================
# normalizer architecture
# ====================================================
model_handle_normalizer = model_zoo.net2D_i2i
norm_kernel_size = 3
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
norm_batch_norm = False

# ====================================================
# settings of the i2l mapper 
# ====================================================
model_handle_i2l = model_zoo.unet2D_i2l

# ====================================================
# settings of the DAE (l2l mapper)
# ====================================================
model_handle_l2l = model_zoo.dae3D

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256)
image_depth = 32
image_size_3D = (image_depth, 256, 256)
loss_type = 'dice'
nlabels = 3
target_resolution = (0.625, 0.625)
downsampling_factor_x = 1
downsampling_factor_y = 1
downsampling_factor_z = 1
max_epochs = 1000
vis_epochs = 100

# ======================================================================
# training settings
# ======================================================================
batch_size = 16
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
continue_run = False
debug = False

# max steps and frequencies for i2i updates
max_steps_i2i = int(image_depth / batch_size)*max_epochs + 1
train_eval_frequency_i2i = int(image_depth / batch_size)*25
vis_frequency_i2i = int(image_depth / batch_size)*vis_epochs

# data aug settings
da_ratio = 0.25
sigma = 20
alpha = 1000
trans_min = -10
trans_max = 10
rot_min = -10
rot_max = 10
scale_min = 0.9
scale_max = 1.1
gamma_min = 0.5
gamma_max = 2.0
brightness_min = 0.0
brightness_max = 0.1
noise_min = 0.0
noise_max = 0.1