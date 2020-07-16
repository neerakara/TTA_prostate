# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import shutil
import tensorflow as tf
import numpy as np
import gc
import model as model
import config.system as sys_config
import sklearn.metrics as met
from skimage.transform import rescale
from tfwrapper import losses, layers
import utils
import utils_vis
import data.data_nci as data_nci
import data.data_promise as data_promise
import data.data_pirad_erc as data_pirad_erc

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
target_resolution = exp_config.target_resolution
image_size = exp_config.image_size
nlabels = exp_config.nlabels
    
# ==================================================================
# main function for training
# ==================================================================
def run_test_time_training(log_dir,
                           image):
    
    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0
        
    # ================================================================
    # reset the graph built so far and build a new TF graph
    # ================================================================
    tf.reset_default_graph()
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ================================================================
        # create placeholders - segmentation net
        # ================================================================
        images_pl = tf.placeholder(tf.float32, shape = [exp_config.batch_size] + list(exp_config.image_size) + [1], name = 'images')        
        labels_dae_pl = tf.placeholder(tf.float32, shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nlabels], name = 'labels_dae')        
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module is trained for each test image
        # ================================================================
        images_normalized, _ = model.normalize(images_pl,
                                               exp_config,
                                               training_pl)         
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        pred_logits, pred_softmax, pred_seg = model.predict_i2l(images_normalized,
                                                                exp_config,
                                                                training_pl = tf.constant(False, dtype=tf.bool))
        
        # ================================================================
        # 3d prior in the label space (DAE)
        # ================================================================
        labels_3d_shape = [1] + list(exp_config.image_size_3D)
        # predict the current segmentation for the entire volume, downsample it and pass it through this placeholder
        pred_seg_3d_pl = tf.placeholder(tf.uint8, shape = labels_3d_shape, name = 'pred_seg_3d')
        pred_seg_3d_1hot_pl = tf.one_hot(pred_seg_3d_pl, depth = exp_config.nlabels)
        
        # denoise the noisy segmentation
        _, pred_seg_3d_denoised_softmax, _ = model.predict_dae(pred_seg_3d_1hot_pl,
                                                               exp_config,
                                                               training_pl = tf.constant(False, dtype=tf.bool))
                
        # ================================================================
        # The loss that will be minimized is the dice between the predictions and the dae outputs 
        # ================================================================        
        loss_op = model.loss(logits = pred_logits,
                             labels = labels_dae_pl,
                             nlabels = exp_config.nlabels,
                             loss_type = 'dice',
                             mask_for_loss_within_mask = None,
                             are_labels_1hot = True)            
        
        # ================================================================
        # split the variables of the different networks
        # ================================================================
        i2l_vars = []
        normalization_vars = []
        dae_vars = []
        
        for v in tf.global_variables():
            var_name = v.name        
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
                i2l_vars.append(v)
            elif 'i2l_mapper' in var_name:
                i2l_vars.append(v)
            elif 'l2l_mapper' in var_name:
                dae_vars.append(v)
                
        # ================================================================
        # create savers
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_dae = tf.train.Saver(var_list = dae_vars)
        saver_test_data = tf.train.Saver(var_list = normalization_vars, max_to_keep=3)        
        saver_best_loss = tf.train.Saver(var_list = normalization_vars, max_to_keep=3)    
        
        # ================================================================
        # add optimization ops
        # ================================================================
        if exp_config.debug: print('creating training op...')
        
        # create an instance of the required optimizer
        optimizer = exp_config.optimizer_handle(learning_rate = exp_config.learning_rate)
        
        # initialize variable holding the accumlated gradients and create a zero-initialisation op
        accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in normalization_vars]
        
        # accumulated gradients init op
        accumulated_gradients_zero_op = [ac.assign(tf.zeros_like(ac)) for ac in accumulated_gradients]

        # calculate gradients and define accumulation op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss_op, var_list = normalization_vars)
            # compute_gradients return a list of (gradient, variable) pairs.
        accumulate_gradients_op = [ac.assign_add(gg[0]) for ac, gg in zip(accumulated_gradients, gradients)]

        # define the gradient mean op
        num_accumulation_steps_pl = tf.placeholder(dtype=tf.float32, name='num_accumulation_steps')
        accumulated_gradients_mean_op = [ag.assign(tf.divide(ag, num_accumulation_steps_pl)) for ag in accumulated_gradients]

        # reassemble the gradients in the [value, var] format and do define train op
        final_gradients = [(ag, gg[1]) for ag, gg in zip(accumulated_gradients, gradients)]
        train_op = optimizer.apply_gradients(final_gradients)

        # ================================================================
        # sequence of running opt ops:
        # 1. at the start of each epoch, run accumulated_gradients_zero_op (no need to provide values for any placeholders)
        # 2. in each training iteration, run accumulate_gradients_op with regular feed dict of inputs and outputs
        # 3. at the end of the epoch (after all batches of the volume have been passed), run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl
        # 4. finally, run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
        # ================================================================

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        
        # ================================================================
        # find if any vars are uninitialized
        # ================================================================
        if exp_config.debug: logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # other summaries 
        # ================================================================        
        loss_dae_3D = tf.placeholder(tf.float32, shape=[], name='loss_dae_3D')
        loss_dae_3D_summary = tf.summary.scalar('test_img/loss_dae_3D', loss_dae_3D)
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        if exp_config.debug:
            logging.info('========================================')
            logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        if exp_config.debug:
            logging.info('========================================')
            logging.info('initializing all variables...')
        sess.run(init_ops)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        uninit_variables = sess.run(uninit_vars)
        if exp_config.debug:
            logging.info('========================================')
            logging.info('This is the list of uninitialized variables:' )
            for v in uninit_variables: print(v)

        # ================================================================
        # Restore the segmentation network parameters and the pre-trained i2i mapper parameters
        # ================================================================
        logging.info('========================================')     
        path_to_model = sys_config.project_code_root + 'models/i2l/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # Restore the DAE parameters
        # ================================================================
        logging.info('========================================')
        path_to_model = sys_config.project_code_root + 'models/l2l/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_dae.restore(sess, checkpoint_path)
               
        # ================================================================
        # run training epochs
        # ================================================================
        step = init_step
        best_loss = 1000.0

        while (step < exp_config.max_steps_i2i):
                                
            # ================================================               
            # After every some epochs, get the dae and vae losses over the entire volume
            # ================================================ 
            if (step == init_step) or (step % exp_config.train_eval_frequency_i2i is 0): 
                
                # ==================
                # compute the current 3d prediction
                # ==================                        
                y_pred_soft = []
                for batch in iterate_minibatches_images(image,
                                                        batch_size = exp_config.batch_size):                    
                    x = batch
                    feed_dict = {images_pl: x, training_pl: False}
                    y_pred_soft.append(sess.run(pred_softmax, feed_dict = {images_pl: x, training_pl: False}))
                
                y_pred_soft = np.squeeze(np.array(y_pred_soft)).astype(float)  
                y_pred_soft = np.reshape(y_pred_soft, [-1, y_pred_soft.shape[2], y_pred_soft.shape[3], y_pred_soft.shape[4]])                
                y_pred = np.argmax(y_pred_soft, axis=-1)

                # ==================
                # denoise the predicted 3D segmentation using the DAE
                # ==================
                feed_dict = {pred_seg_3d_1hot_pl: np.expand_dims(y_pred_soft, axis=0)}                 
                y_pred_noisy_denoised_softmax = np.squeeze(sess.run(pred_seg_3d_denoised_softmax, feed_dict=feed_dict)).astype(np.float16)               
                y_pred_noisy_denoised = np.argmax(y_pred_noisy_denoised_softmax, axis=-1)
                loss_dae = 1 - np.mean(met.f1_score(y_pred.flatten(), y_pred_noisy_denoised.flatten(), average=None)[1:])
                
                # ==================
                # update losses on tensorboard
                # ==================
                summary_writer.add_summary(sess.run(loss_dae_3D_summary, feed_dict={loss_dae_3D: loss_dae}), step)
                
                # ==================
                # save best model so far
                # ==================
                if best_loss > loss_dae:
                    best_loss = loss_dae
                    best_file = os.path.join(log_dir, 'models/best_loss.ckpt')
                    saver_best_loss.save(sess, best_file, global_step=step)
                    logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

                if step % exp_config.vis_frequency_i2i is 0:
                    # ==================
                    # save checkpoint
                    # ==================
                    logging.info('=============== Saving checkkpoint at step %d ... ' % step)
                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver_test_data.save(sess, checkpoint_file, global_step=step)
                        
                    # ==================
                    # visualize results
                    # ==================                                        
                    x_norm = []
                    for batch in iterate_minibatches_images(image,
                                                            batch_size = exp_config.batch_size):
                        x = batch
                        x_norm.append(sess.run(images_normalized, feed_dict = {images_pl: x, training_pl: False}))                        
                    x_norm = np.squeeze(np.array(x_norm)).astype(float)  
                    x_norm = np.reshape(x_norm, [-1, x_norm.shape[2], x_norm.shape[3]])
                    
                    utils_vis.save_sample_results(x = image,
                                                  x_norm = x_norm,
                                                  y_pred = y_pred,
                                                  y_pred_dae = y_pred_noisy_denoised,
                                                  savepath = log_dir + '/results/step' + str(step) + '.png')

            # ================================================     
            # Part of training ops sequence:
            # 1. At the start of each epoch, run accumulated_gradients_zero_op (no need to provide values for any placeholders)
            # ================================================               
            sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0
                
            # ================================================               
            # batches
            # ================================================    
            for batch in iterate_minibatches_images_and_labels(image,
                                                               y_pred_noisy_denoised_softmax,
                                                               exp_config.batch_size):

                x, y = batch   
                
                # ===========================
                # define feed dict for this iteration
                # ===========================   
                feed_dict = {images_pl: x,
                             labels_dae_pl: y,
                             training_pl: True}
                
                # ================================================     
                # Part of training ops sequence:
                # 2. in each training iteration, run accumulate_gradients_op with regular feed dict of inputs and outputs
                # ================================================               
                sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                num_accumulation_steps = num_accumulation_steps + 1
                
                step += 1
                
            # ================================================     
            # Part of training ops sequence:
            # 3. At the end of the epoch (after all batches of the volume have been passed), run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl
            # ================================================     
            sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
                    
            # ================================================================
            # sequence of running opt ops:
            # 4. finally, run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
            # ================================================================    
            sess.run(train_op, feed_dict=feed_dict)            
                
        # ================================================================    
        # ================================================================    
        sess.close()

    # ================================================================      
    # ================================================================    
    gc.collect()
    
    return 0
        
# ==================================================================
# ==================================================================
def iterate_minibatches_images(images,
                               batch_size):
        
    images_ = np.copy(images)
    
    # generate indices to randomly select subjects in each minibatch
    n_images = images_.shape[0]
    random_indices = np.arange(n_images)

    # generate batches in a for loop
    for b_i in range(n_images // batch_size):
        if b_i + batch_size > n_images:
            continue
        batch_indices = random_indices[b_i*batch_size:(b_i+1)*batch_size]
        images_this_batch = np.expand_dims(images_[batch_indices, ...], axis=-1)

        yield images_this_batch
                    
# ==================================================================
# ==================================================================
def iterate_minibatches_images_and_labels(images,
                                          labels,
                                          batch_size):
        
    images_ = np.copy(images)
    labels_ = np.copy(labels)
    
    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images_.shape[0] # 32
    random_indices = np.arange(n_images)

    # ===========================
    # generate batches in a for loop
    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue

        images_this_batch = np.expand_dims(images_[random_indices[b_i*batch_size:(b_i+1)*batch_size], ...], axis=-1)
        labels_this_batch = labels_[random_indices[b_i*batch_size:(b_i+1)*batch_size], ...]

        yield images_this_batch, labels_this_batch

# ==================================================================
# ==================================================================
def main(argv):
        
    # ============================
    # Load test image
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')    
    if exp_config.test_dataset is 'PIRAD_ERC':

        image_depth = 32
        z_resolution = 2.5

        data_pros = data_pirad_erc.load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                             preproc_folder = sys_config.preproc_folder_pirad_erc,
                                             idx_start = 0,
                                             idx_end = 20,
                                             size = image_size,
                                             target_resolution = target_resolution,
                                             labeller = 'ek')
        
        imts, gtts = [data_pros['images'], data_pros['labels']]
        num_slices_in_test_subjects = data_pros['nz'][:]
        name_test_subjects = data_pros['patnames']
        slice_thickness_in_test_subjects = data_pros['pz'][:]
        
    # ================================================================
    # create a text file for writing results
    # results of individual subjects will be appended to this file
    # ================================================================
    log_dir_base = sys_config.project_code_root + 'models/i2i/'
    if not tf.gfile.Exists(log_dir_base):
        tf.gfile.MakeDirs(log_dir_base)
    
    # ================================================================
    # run the training for each test image
    # ================================================================
    subject_num = int(argv[0])
    for subject_id in range(subject_num, subject_num+1):
        
        subject_id_start_slice = np.sum(num_slices_in_test_subjects[:subject_id])
        subject_id_end_slice = np.sum(num_slices_in_test_subjects[:subject_id+1])
        s_vis = 1 # required to define which slices to visualize

        image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        label = gtts[subject_id_start_slice:subject_id_end_slice,:,:] 
        label = label.astype(np.uint8)
        slice_thickness_this_subject = slice_thickness_in_test_subjects[subject_id]
                
        # ==================================================================
        # setup logging
        # ==================================================================
        subject_name = str(name_test_subjects[subject_id])[2:-1]
        log_dir = log_dir_base + '/subject_' + subject_name
        logging.info('============================================================')
        logging.info('Logging directory: %s' %log_dir)
        logging.info('Subject ID: %d' %subject_id)
        logging.info('Subject name: %s' %subject_name)
        
        # ===========================
        # create dir if it does not exist
        # ===========================
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)
            tf.gfile.MakeDirs(log_dir + '/models')
            tf.gfile.MakeDirs(log_dir + '/results')
            
        # ===========================
        # Copy experiment config file
        # ===========================
        shutil.copy(exp_config.__file__, log_dir)
        
        # visualize image and ground truth label
        utils_vis.save_samples_downsampled(utils.crop_or_pad_volume_to_size_along_x(image, image_depth)[::s_vis, :, :],
                                           savepath = log_dir + '/orig_image.png', add_pixel_each_label=False, cmap='gray')
        utils_vis.save_samples_downsampled(utils.crop_or_pad_volume_to_size_along_x(label, image_depth)[::s_vis, :, :],
                                           savepath = log_dir + '/gt_label.png', cmap='tab20')

        # ===========================
        # Change the resolution of the current image so that it matches the atlas, and pad and crop.
        # ===========================
        image_rescaled_cropped, label_rescaled_cropped = rescale_image_and_label(image,
                                                                                 label,
                                                                                 slice_thickness_this_subject,
                                                                                 new_resolution = z_resolution,
                                                                                 new_depth = image_depth)
        
        # visualize rescaled image and ground truth label
        utils_vis.save_samples_downsampled(image_rescaled_cropped[::s_vis, :, :], savepath = log_dir + '/orig_image_rescaled.png', add_pixel_each_label=False, cmap='gray')
        utils_vis.save_samples_downsampled(label_rescaled_cropped[::s_vis, :, :], savepath = log_dir + '/gt_label_rescaled.png', cmap='tab20')
        
        # ===========================
        # run test time training for this subject
        # ===========================        
        run_test_time_training(log_dir,
                               image_rescaled_cropped)
        
        # ===========================
        # ===========================
        gc.collect()
        
# ===========================================================================
# ===========================================================================
def rescale_image_and_label(image,
                            label,
                            slice_thickness_this_subject,
                            new_resolution,
                            new_depth):
    
    image_rescaled = []
    label_rescaled = []
            
    # ======================
    # rescale in 3d
    # ======================
    scale_vector = [slice_thickness_this_subject / new_resolution, # for this axes, the resolution was kept unchanged during the initial 2D data preprocessing. but for the atlas (made from hcp labels), all of them have 0.7mm slice thickness
                    1.0, # the resolution along these 2 axes was made as required in the initial 2d data processing already
                    1.0]
    
    image_rescaled = rescale(image,
                             scale_vector,
                             order=1,
                             preserve_range=True,
                             multichannel=False,
                             mode = 'constant')

    label_onehot = utils.make_onehot(label, exp_config.nlabels)

    label_onehot_rescaled = rescale(label_onehot,
                                    scale_vector,
                                    order=1,
                                    preserve_range=True,
                                    multichannel=True,
                                    mode='constant')
    
    label_rescaled = np.argmax(label_onehot_rescaled, axis=-1)
        
    # =================
    # crop / pad
    # =================
    image_rescaled_cropped = utils.crop_or_pad_volume_to_size_along_x(image_rescaled, new_depth).astype(np.float32)
    label_rescaled_cropped = utils.crop_or_pad_volume_to_size_along_x(label_rescaled, new_depth).astype(np.uint8)
            
    return image_rescaled_cropped, label_rescaled_cropped

# ==================================================================
# ==================================================================
import sys
if __name__ == "__main__":
    main(sys.argv[1:])