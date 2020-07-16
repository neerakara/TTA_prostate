# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import model as model
import config.system as sys_config

import data.data_nci as data_nci
import data.data_promise as data_promise
import data.data_pirad_erc as data_pirad_erc

from skimage.transform import rescale
import sklearn.metrics as met

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
if exp_config.normalize is True:
    log_dir = sys_config.project_code_root + 'models/i2i/'
else:
    log_dir = sys_config.project_code_root + 'models/i2l/'

target_resolution = exp_config.target_resolution
image_size = exp_config.image_size
nlabels = exp_config.nlabels

# ==================================================================
# main function for inference
# ==================================================================
def predict_segmentation(subject_name,
                         image,
                         normalize = True):
    
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ================================================================
        # create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32,
                                   shape = [None] + list(image_size) + [1],
                                   name = 'images')

        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module is trained for each test image
        # ================================================================
        images_normalized, added_residual = model.normalize(images_pl,
                                                            exp_config,
                                                            training_pl = tf.constant(False, dtype=tf.bool))
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, softmax, preds = model.predict_i2l(images_normalized,
                                                   exp_config,
                                                   training_pl = tf.constant(False, dtype=tf.bool))
                        
        # ================================================================
        # divide the vars into segmentation network and normalization network
        # ================================================================
        i2l_vars = []
        normalization_vars = []
        
        for v in tf.global_variables():
            var_name = v.name        
            i2l_vars.append(v)
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
                                
        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
                
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_normalizer = tf.train.Saver(var_list = normalization_vars)        
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        sess.run(init_ops)
        
        # ================================================================
        # Restore the segmentation network parameters
        # ================================================================
        logging.info('============================================================')   
        path_to_model = sys_config.project_code_root + 'models/i2l/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # Restore the normalization network parameters
        # ================================================================
        if normalize is True:
            logging.info('============================================================')
            path_to_model = sys_config.project_code_root + 'models/i2i/subject_' + subject_name + '/models/'
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_loss.ckpt')
            logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
            saver_normalizer.restore(sess, checkpoint_path)
            logging.info('============================================================')

        # ================================================================
        # Make predictions for the image at the resolution of the image after pre-processing
        # ================================================================
        label_predicted = []
        image_normalized = []
        
        for b_i in range(0, image.shape[0], 1):
        
            X = np.expand_dims(image[b_i:b_i+1, ...], axis=-1)
            
            label_predicted.append(sess.run(preds, feed_dict={images_pl: X}))
            image_normalized.append(sess.run(images_normalized, feed_dict={images_pl: X}))
        
        label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  
        image_normalized = np.squeeze(np.array(image_normalized)).astype(float)  
        
        sess.close()
        
        return label_predicted, image_normalized
    
# ================================================================
# ================================================================
def rescale_and_crop(arr,
                     px,
                     py,
                     nx,
                     ny,
                     order_interpolation,
                     num_rotations):
    
    # 'target_resolution_brain' contains the resolution that the images were rescaled to, during the pre-processing.
    # we need to undo this rescaling before evaluation
    scale_vector = [target_resolution[0] / px,
                    target_resolution[1] / py]

    arr_list = []
    
    for zz in range(arr.shape[0]):
     
        # ============
        # rotate the labels back to the original orientation
        # ============            
        arr2d_rotated = np.rot90(np.squeeze(arr[zz, :, :]), k=num_rotations)
        
        arr2d_rescaled = rescale(arr2d_rotated,
                                 scale_vector,
                                 order = order_interpolation,
                                 preserve_range = True,
                                 multichannel = False,
                                 mode = 'constant')

        arr2d_rescaled_cropped = utils.crop_or_pad_slice_to_size(arr2d_rescaled, nx, ny)

        arr_list.append(arr2d_rescaled_cropped)
    
    arr_orig_res_and_size = np.array(arr_list)
    arr_orig_res_and_size = arr_orig_res_and_size.swapaxes(0, 1).swapaxes(1, 2)
    
    return arr_orig_res_and_size
        
# ==================================================================
# ==================================================================
def main():
    
    # ===================================
    # read the test images
    # ===================================
    test_dataset_name = exp_config.test_dataset
    
    if test_dataset_name is 'NCI':
        data_pros = data_nci.load_and_maybe_process_data(input_folder=sys_config.orig_data_root_nci,
                                                         preprocessing_folder=sys_config.preproc_folder_nci,
                                                         size=image_size,
                                                         target_resolution=target_resolution,
                                                         force_overwrite=False,
                                                         cv_fold_num = 1)

        imts = data_pros['images_test']
        name_test_subjects = data_pros['patnames_test']

        orig_data_res_x = data_pros['px_test'][:]
        orig_data_res_y = data_pros['py_test'][:]
        orig_data_res_z = data_pros['pz_test'][:]
        orig_data_siz_x = data_pros['nx_test'][:]
        orig_data_siz_y = data_pros['ny_test'][:]
        orig_data_siz_z = data_pros['nz_test'][:]

        num_test_subjects = orig_data_siz_z.shape[0]
        ids = np.arange(num_test_subjects)

    elif test_dataset_name is 'PIRAD_ERC':

        idx_start = 0
        idx_end = 20
        ids = np.arange(idx_start, idx_end)

        data_pros = data_pirad_erc.load_data(input_folder=sys_config.orig_data_root_pirad_erc,
                                             preproc_folder=sys_config.preproc_folder_pirad_erc,
                                             idx_start=idx_start,
                                             idx_end=idx_end,
                                             size=image_size,
                                             target_resolution=target_resolution,
                                             labeller='ek')
        imts = data_pros['images']
        name_test_subjects = data_pros['patnames']

        orig_data_res_x = data_pros['px'][:]
        orig_data_res_y = data_pros['py'][:]
        orig_data_res_z = data_pros['pz'][:]
        orig_data_siz_x = data_pros['nx'][:]
        orig_data_siz_y = data_pros['ny'][:]
        orig_data_siz_z = data_pros['nz'][:]

        num_test_subjects = orig_data_siz_z.shape[0]
            
    # ================================   
    # open a text file for writing the mean dice scores for each subject that is evaluated
    # ================================
    if exp_config.whole_gland_results is True:
        results_file = open(log_dir + test_dataset_name + '_test_whole_gland.txt', "w")
    else:
        results_file = open(log_dir + test_dataset_name + '_test.txt', "w")
    results_file.write("================================== \n") 
    results_file.write("Test results \n") 
    
    # ================================================================
    # For each test image, load the best model and compute the dice with this model
    # ================================================================
    dice_per_label_per_subject = []
    hsd_per_label_per_subject = []

    for sub_num in range(1):#(num_test_subjects):

        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
        image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        subject_name = str(name_test_subjects[sub_num])[2:-1]
        logging.info('============================================================')
        logging.info('Subject id: %s' %sub_num)
    
        # ==================================================================
        # predict segmentation at the pre-processed resolution
        # ==================================================================
        predicted_labels, normalized_image = predict_segmentation(subject_name,
                                                                  image,
                                                                  exp_config.normalize)
                
        # ==================================================================
        # read the original segmentation mask
        # ==================================================================
        if test_dataset_name is 'NCI':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_nci.load_without_size_preprocessing(sys_config.orig_data_root_nci,
                                                                               cv_fold_num=1,
                                                                               train_test='test',
                                                                               idx=ids[sub_num])
            num_rotations = 0

        elif test_dataset_name is 'PIRAD_ERC':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_pirad_erc.load_without_size_preprocessing(sys_config.orig_data_root_pirad_erc,
                                                                                     ids[sub_num],
                                                                                     labeller='ek')
            num_rotations = -3
            
        # ==================================================================
        # convert the predicitons back to original resolution
        # ==================================================================
        predicted_labels_orig_res_and_size = rescale_and_crop(predicted_labels,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 0,
                                                              num_rotations = num_rotations)

        # ==================================================================
        # If only whole-gland comparisions are desired, merge the labels in both ground truth segmentations as well as the predictions
        # ==================================================================
        if exp_config.whole_gland_results is True:
            predicted_labels_orig_res_and_size[predicted_labels_orig_res_and_size!=0] = 1
            labels_orig[labels_orig!=0] = 1
            nl = 2
            savepath = log_dir + test_dataset_name + '_test_' + subject_name + '_whole_gland.png'
        else:
            nl = nlabels
            savepath = log_dir + test_dataset_name + '_test_' + subject_name + '.png'

        # ==================================================================
        # compute dice at the original resolution
        # ==================================================================    
        dice_per_label_this_subject = met.f1_score(labels_orig.flatten(),
                                                   predicted_labels_orig_res_and_size.flatten(),
                                                   average=None)
        
        # ==================================================================    
        # compute Hausforff distance at the original resolution
        # ==================================================================   
        compute_hsd = False
        if compute_hsd is True:
            hsd_per_label_this_subject = utils.compute_surface_distance(y1 = labels_orig,
                                                                        y2 = predicted_labels_orig_res_and_size,
                                                                        nlabels = nl)
        else:
            hsd_per_label_this_subject = np.zeros((nl))
        
        # ================================================================
        # save sample results
        # ================================================================
        d_vis = 32
        ids_vis = np.arange(0, 32, 4)
        utils_vis.save_sample_prediction_results(x = utils.crop_or_pad_volume_to_size_along_z(image_orig, d_vis),
                                                 x_norm = utils.crop_or_pad_volume_to_size_along_z(image_orig, d_vis),
                                                 y_pred = utils.crop_or_pad_volume_to_size_along_z(predicted_labels_orig_res_and_size, d_vis),
                                                 gt = utils.crop_or_pad_volume_to_size_along_z(labels_orig, d_vis),
                                                 num_rotations = - num_rotations, # rotate for consistent visualization across datasets
                                                 savepath = savepath,
                                                 nlabels = nl,
                                                 ids=ids_vis)
                                   
        # ================================
        # write the mean fg dice of this subject to the text file
        # ================================
        results_file.write(subject_name + ":: dice (mean, std over all FG labels): ")
        results_file.write(str(np.round(np.mean(dice_per_label_this_subject[1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)))
        results_file.write(", hausdorff distance (mean, std over all FG labels): ")
        results_file.write(str(np.round(np.mean(hsd_per_label_this_subject), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)) + "\n")
        
        dice_per_label_per_subject.append(dice_per_label_this_subject)
        hsd_per_label_per_subject.append(hsd_per_label_this_subject)
    
    # ================================================================
    # write per label statistics over all subjects    
    # ================================================================
    dice_per_label_per_subject = np.array(dice_per_label_per_subject)
    hsd_per_label_per_subject =  np.array(hsd_per_label_per_subject)
    
    # ================================
    # In the array images_dice, in the rows, there are subjects
    # and in the columns, there are the dice scores for each label for a particular subject
    # ================================
    results_file.write("================================== \n") 
    results_file.write("Label: dice mean, std. deviation over all subjects\n")
    for i in range(dice_per_label_per_subject.shape[1]):
        results_file.write(str(i) + ": " + str(np.round(np.mean(dice_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,i]), 3)) + "\n")
    results_file.write("================================== \n") 
    results_file.write("Label: hausdorff distance mean, std. deviation over all subjects\n")
    for i in range(hsd_per_label_per_subject.shape[1]):
        results_file.write(str(i+1) + ": " + str(np.round(np.mean(hsd_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject[:,i]), 3)) + "\n")
    
    # ==================
    # write the mean dice over all subjects and all labels
    # ==================
    results_file.write("================================== \n") 
    results_file.write("DICE Mean, std. deviation over foreground labels over all subjects: " + str(np.round(np.mean(dice_per_label_per_subject[:,1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,1:]), 3)) + "\n")
    results_file.write("HSD Mean, std. deviation over labels over all subjects: " + str(np.round(np.mean(hsd_per_label_per_subject), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject), 3)) + "\n")
    results_file.write("================================== \n") 
    results_file.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()