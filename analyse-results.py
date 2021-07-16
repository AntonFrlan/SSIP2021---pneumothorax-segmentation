# Analyse the output of the U-Net
# Classification: ROC, AUC, Confusion matrix
# Segmentation: Dice index, IoU

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import glob
import os

# Define a function to calculate the Dice index
def calculate_dice_index(im1, im2):

	# Convert the images to bool
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    # Check if both masks are all zero and flip the values
    if not np.any(im1) and not np.any(im2):
        im1 = np.logical_not(im1)
        im2 = np.logical_not(im2)

    # Raise a warning if the images are misshapen
    if im1.shape != im2.shape:
        raise ValueError('Shape mismatch: im1 and im2 must have the same shape.')

    # Compute the Dice coefficient
    intersection = np.logical_and(im1, im2)
    dice = 2. * intersection.sum() / (im1.sum() + im2.sum())

    # Print the dice score
    #print('Dice index: {}'.format(dice))

    return dice

# Define a function to calculate the IoU (intersection over union)
def calculate_iou(im1, im2):

	# Convert the images to bool
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    # Check if both masks are all zero and flip the values
    if not np.any(im1) and not np.any(im2):
        im1 = np.logical_not(im1)
        im2 = np.logical_not(im2)

    # Caluclate IoU
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    iou = np.sum(intersection) / np.sum(union)

	# Print the IoU score
    #print('IoU: {}'.format(iou))

    return iou

def plot_labels_and_predictions(im1, im2):
	# Convert images to bitmap
    im1[im1 > 0] = 1
    im2[im2 > 0] = 1
    # Plot the images
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Label')
    plt.imshow(im1, cmap=plt.cm.bone)
    plt.subplot(1, 3, 2)
    plt.title('Prediction')
    plt.imshow(im2, cmap=plt.cm.bone)
    plt.subplot(1, 3, 3)
    plt.title('Intersection')
    plt.imshow(np.logical_and(im1, im2), cmap=plt.cm.bone)
    plt.show()

def calulculate_roc_and_auc(vals_true, vals_pred):
    fpr, tpr, thresholds = roc_curve(vals_true, vals_pred)
    auc = roc_auc_score(vals_true, vals_pred)
    plt.plot(fpr, tpr, '-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.text(0.7, 0.125, r'AUC = %.3f' %auc, bbox = {'facecolor':'white', 'alpha':0.2, 'pad':5})
    plt.show()

def calculate_and_plot_confusion_matrix(vals_true, vals_pred):
    # Chech if the input is a list or a numpy array
    if((isinstance(vals_true, np.ndarray) & isinstance(vals_pred, np.ndarray))):
        # Preallocate the matrices
        vals_true_temp = []
        vals_pred_temp = []
        # Iterate through all values
        for i in range(len(vals_true)):
            if(vals_true[i] == 1):
                vals_true_temp.append('Pneumothorax')
                vals_pred_temp.append('Pneumothorax')
            else:
                vals_true_temp.append('Normal')
                vals_pred_temp.append('Normal')
        #print(vals_true_temp)
        #print(vals_pred_temp)
    else:
        print('List')
    
    # Assign temporary arrays to the original ones
    vals_true = vals_true_temp
    vals_pred = vals_pred_temp
    
    # Plot the confusion matrix   
    conf_mat = confusion_matrix(vals_true, vals_pred, labels=['Normal', 'Pneumothorax']).tolist()
    df_cm = pd.DataFrame(conf_mat, ['Normal', 'Pneumothorax'], ['Normal', 'Pneumothorax'])
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 12}) # font size
    #plt.title('Confusion matrix')
    plt.xlabel('True labels', fontsize = '12', fontweight = 'bold')
    plt.ylabel('Predicted labels', fontsize = '12', fontweight = 'bold')
    plt.yticks(np.arange(2)+0.5,('Normal', 'Pneumothorax'), rotation=90, va='center')
    plt.show()

def extract_true_vals(path_to_csv_table):

    # Extract the data from the accompanying CSV file
    data_test = pd.read_csv(path_to_csv_table, delimiter=',')
    uid = data_test['ImageId'].to_numpy()
    pixels = data_test[' EncodedPixels'].to_numpy()

    # Preallocate the ground truth labels array
    true_labels = np.ones((pixels.shape))

    # Iterate through the pixels to extract the true labels
    for i in range(len(pixels)):
        if pixels[i] == '-1':
            true_labels[i] = -1
        else:
            continue

    # Find the unique values 
    [unique_true_uid, unique_true_uid_ind] = np.unique(uid, return_index = True)
    unique_true_labels = true_labels[unique_true_uid_ind]

    return [unique_true_uid, unique_true_labels]

def extract_pred_vals(path_to_pred_labels):

    # Get the paths to the data
    pred_labels_path = glob.glob(path_to_pred_labels)

    # Preallocate the predicted labels array
    pred_labels = np.zeros(len(pred_labels_path))

    # Extract the uids
    uid = [os.path.basename(path[:-4]) for path in pred_labels_path]

    # Iterate through all predicted labels
    for i in range(len(pred_labels_path)):
        # Load the labelled image
        temp_pred_labels = np.load(pred_labels_path[i])
        # Convert the labelled image to bitmap
        temp_pred_labels[temp_pred_labels > 0] = 1
        # Check if the image has any non-zero values
        if(temp_pred_labels.any()):
            pred_labels[i] = 1
        else:
            pred_labels[i] = -1

    return [uid, pred_labels]

def compare_true_and_pred_vals(true_vals, pred_vals, true_uid, pred_uid):

    # Preallocate the matrix of used ground truth images (masks)
    used_true_vals = np.zeros(len(pred_uid))

    # Iterate through the predictions and find the used true values
    for i in range(len(pred_uid)):
        for j in range(len(true_uid)):
            if(pred_uid[i] == true_uid[j]):
                used_true_vals[i] = true_vals[j]
            else:
                continue

    return used_true_vals

def calculate_dice_and_iou_for_all_cases(path_to_true_labels, path_to_pred_labels):

    # Get the paths to the data
    true_labels_path = glob.glob(path_to_true_labels)
    pred_labels_path = glob.glob(path_to_pred_labels)

    # Preallocate the arrays for Dice index and the IoU
    dice = np.zeros(len(pred_labels_path))
    iou = np.zeros(len(pred_labels_path))

    # Iterate through all the labels
    for i in range(len(pred_labels_path)):
        # Load the ground truth label and the prediction
        temp_img_true = np.load(true_labels_path[i])
        temp_img_pred = np.load(pred_labels_path[i])
        # Calculate the dice coefficient
        dice[i] = calculate_dice_index(temp_img_true, temp_img_pred)
        iou[i] = calculate_iou(temp_img_true, temp_img_pred)

    return [dice, iou]

# The main script 
if __name__ == "__main__":
    # Extract the true labels from the CSV file
    [true_labels_uid, true_labels_vals] = extract_true_vals('path_to_csv')
    # Extract the predicted labels from the saved files
    [pred_labels_uid, pred_labels_vals] = extract_pred_vals('path_to_predicted_labels/*.npy')
    # Find the used true labels
    used_true_labels_vals = compare_true_and_pred_vals(true_labels_vals, pred_labels_vals, true_labels_uid, pred_labels_uid)

    for i in range(len(used_true_labels_vals)):
        if used_true_labels_vals[i] == 0:
            used_true_labels_vals[i] = -1
    #print(np.unique(used_true_labels_vals))
    # Plot the confusion matrix
    calculate_and_plot_confusion_matrix(used_true_labels_vals, pred_labels_vals)
    # Plot the ROC curve and calculate the AUC value
    calulculate_roc_and_auc(used_true_labels_vals, pred_labels_vals)
    # Iterate through all image masks to calculate the Dice index and the IoU
    dice, iou = calculate_dice_and_iou_for_all_cases('path_to_true_labels/*.npy', 'path_to_predicted_labels/*.npy')
    # Save the results as a numpy array
    np.save('dice_index.npy', dice)
    np.save('iou.npy', iou)