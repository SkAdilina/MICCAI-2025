import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from utils.helpers import *

losses_used = "/exp_longit_CFSegNet"
task_path = "PATH_TO_TASK_DIR/Task_FreeSurfer_1-5T_longit"
task_name_exp = "Longit_CFSegNet"
print("-------", task_name_exp, "--- in criterion.py----")

file_loc = task_path + "/Demographics-ADNI-1-5T-MNI-ICV.csv"
demographics = pd.read_csv(file_loc)[['ID', 'TP', 'Age', 'Diagnosis', 'ICV']] 
max_tp = 13

# Define the mapping of strings to numbers
mapping = {
    "Healthy control": 1,
    "MCI patient": 2,
    "AD patient": 3,
    "Fronto-temporal Dementia": -1,
    "Primary Progressive Aphasia": -1,
    "Other": -1
}

#epsilon = torch.tensor(1e-8, dtype=torch.float32)
epsilon = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#Equation for AD patient in Hippocampus_Adj:  + x^1 + x^2

def clear_gpu_memory():
    torch.cuda.empty_cache()

def healthy_control(x):
    return -1.088e+00 + 2.579e-01 * x - 1.587e-02 * x**2

def mci_patient(x):
    return -1.506e+00 +  3.381e-01 * x - 1.949e-02 * x**2

def ad_patient(x):
    return -7.694e-01 + 2.965e-02 * x + 8.428e-03 * x**2

def plotting_vol_loss(hippocampus_adj_change, hippocampus_adj_mean, diagnosis_np, cur_fold):

    #print("fold while plotting data points curve :", cur_fold)
    # Convert to NumPy array
    #diagnosis_np = diagnosis.numpy()
    
    #print("hippocampus_adj_change type:", type(hippocampus_adj_change))
    #print("hippocampus_adj_mean type:", type(hippocampus_adj_mean))
    #print("diagnosis type:", type(diagnosis_np))
    #print("diagnosis vals:", diagnosis)

    # Define the range of x values
    x_values = np.linspace(1, 10, 400)  # Adjust the range as needed
    
    # Evaluate the polynomials for each x value
    y_healthy = healthy_control(x_values)
    y_mci = mci_patient(x_values)
    y_ad = ad_patient(x_values)
    
    # Flatten the arrays
    hippocampus_adj_change_flat = hippocampus_adj_change.flatten()
    hippocampus_adj_mean_flat = hippocampus_adj_mean.flatten()
    diagnosis_flat = diagnosis_np.flatten()
    
    # Set up color map for diagnosis values
    colors = {-1: 'black', 1: 'green', 2: 'orange', 3: 'red'}
    
    # Create a list of colors based on diagnosis values, ignoring -1
    point_colors = [colors[val] for val in diagnosis_flat]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_healthy, label="Healthy Control", color="green", linewidth=2)
    plt.plot(x_values, y_mci, label="MCI Patient", color="orange", linewidth=2)
    plt.plot(x_values, y_ad, label="AD Patient", color="red", linewidth=2)
    
    # Add scatter plot for hippocampus_adj_change and hippocampus_adj_mean
    plt.scatter(hippocampus_adj_mean_flat, hippocampus_adj_change_flat, color=point_colors, label="Data Points", alpha=0.3)
    
    # Customize the plot
    plt.xlabel("Mean Volume (Hippocampus_Adj)")
    plt.ylabel("Volume Change per Year (Hippocampus_Adj)")
    plt.title("Polynomial Fits and Data Points")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #plt.savefig(task_path + losses_used + '/vol_loss_plots/' + f"fold_{cur_fold}_plot_{current_time}.png")
    
    #plt.show()
    plt.close() 

    return


def vol_loss_calculate_old_tensor(get_volume, total_tp, ages, ICVs, diagnosis, cur_fold):
    
    # Load the model parameters from the file
    with open('PATH_TO_DIR/model_params.pkl', 'rb') as f:
        params, mean_val = pickle.load(f)
    
    loss = []
    
    # Move get_volume, total_tp, ages, ICVs, diagnosis to GPU
    get_volume = torch.tensor(get_volume, dtype=torch.float32, device='cuda')
    total_tp = torch.tensor(total_tp, dtype=torch.int64, device='cuda')
    ages = torch.tensor(ages, dtype=torch.float32, device='cuda')
    ICVs = torch.tensor(ICVs, dtype=torch.float32, device='cuda')
    diagnosis = torch.tensor(diagnosis, dtype=torch.int64, device='cuda')
    #print("DIAGNOSIS in function tensor", diagnosis)
    
    hippocampus = get_volume[:, :, 1] + get_volume[:, :, 2]
    hippocampus_adj = torch.zeros_like(hippocampus, device='cuda')
    hippocampus_mean = torch.zeros_like(hippocampus, device='cuda')
    vol_diff = torch.zeros_like(hippocampus, device='cuda')
    age_diff = torch.zeros_like(ages, device='cuda')

    for b in range(0, total_tp.shape[0], 1):
        cur_total_tp = total_tp[b]
        
        for i in range(0, cur_total_tp, 1):
            predicted_volume = params['Intercept'] + params['ICV'] * ICVs[b][i]
            hippocampus_adj[b][i] = (hippocampus[b][i] - predicted_volume + mean_val) / 1000

        for i in range(0, cur_total_tp - 1, 1):
            hippocampus_mean[b][i] = (hippocampus_adj[b][i] + hippocampus_adj[b][i + 1]) / 2
            age_diff[b][i] = ages[b][i + 1] - ages[b][i]
            vol_diff[b][i] = (hippocampus[b][i] - hippocampus[b][i + 1]) / 100

    hippocampus_adj_change = vol_diff / age_diff
    plotting_vol_loss(hippocampus_adj_change.cpu().numpy(), hippocampus_mean.cpu().numpy(), diagnosis.cpu().numpy(), cur_fold)
    
    del hippocampus, vol_diff, age_diff
    clear_gpu_memory()

    batch_loss = []
    for b in range(0, total_tp.shape[0], 1):
        cur_loss = []
        distances = 0
        cur_total_tp = total_tp[b] 
        for i in range(0, cur_total_tp - 1, 1):
            if diagnosis[b][i] == 1:
                distances = torch.abs(hippocampus_adj_change[b][i] - healthy_control(hippocampus_mean[b][i]))
            elif diagnosis[b][i] == 2:
                distances = torch.abs(hippocampus_adj_change[b][i] - mci_patient(hippocampus_mean[b][i]))
            elif diagnosis[b][i] == 3:
                distances = torch.abs(hippocampus_adj_change[b][i] - ad_patient(hippocampus_mean[b][i]))
            cur_loss.append(distances)
    
        batch_loss.append(torch.mean(torch.tensor(cur_loss, dtype=torch.float32, device='cuda')))
        
        del cur_loss, distances
        clear_gpu_memory()

    if torch.isnan(torch.tensor(batch_loss)).any():
        print("batch_loss contains NaN values")
        loss = torch.tensor(0.0, requires_grad=True, device='cuda') + epsilon
    else:    
        batch_loss_tensor = torch.tensor(batch_loss, dtype=torch.float32, device='cuda')
        loss = torch.mean(batch_loss_tensor)
        loss = loss + torch.tensor(0.0, requires_grad=True, device='cuda')
        del batch_loss_tensor
        clear_gpu_memory()


    print("Final Vol Loss in old (tensor version)", loss)
    return loss


def smoothness_updated_tensor(volume, total_tp, ages, roi_val, epsilon: float = 1e-6):
    """
    Differentiable smoothness loss for longitudinal volumes based on second-order volume change.
    
    Args:
        volume (Tensor): [B, T, R] tensor of volumes, where R is number of ROIs.
        total_tp (Tensor): [B] tensor with number of valid timepoints per subject in the batch.
        ages (Tensor): [B, T] tensor with age values per timepoint.
        roi_val (int): ROI index to extract for smoothness loss.
        epsilon (float): Small constant to maintain differentiability when loss is 0.

    Returns:
        loss (Tensor): Smoothness loss (scalar).
    """

    # Move get_volume, total_tp, ages to GPU
    volume = torch.tensor(volume, dtype=torch.float32, device='cuda')
    total_tp = torch.tensor(total_tp, dtype=torch.int64, device='cuda')
    ages = torch.tensor(ages, dtype=torch.float32, device='cuda')
    
    B, T, R = volume.shape
    loss_list = []

    for b in range(B):
        cur_tp = total_tp[b].item()

        if cur_tp < 3:
            continue  # Need at least 3 time points for smoothness constraint

        subj_loss = []

        # Extract ROI-specific volume and age for current subject
        vol_roi = volume[b, :cur_tp, roi_val]  # [T]
        age_seq = ages[b, :cur_tp]             # [T]

        # Fill NaNs in age tensor with linear interpolation (still differentiable)
        age_seq = age_seq.clone()
        nan_mask = torch.isnan(age_seq)
        if nan_mask.any():
            valid_idx = (~nan_mask).nonzero().squeeze(-1)
            age_seq[nan_mask] = torch.interp(
                nan_mask.nonzero().float().squeeze(-1),
                valid_idx.float(),
                age_seq[valid_idx]
            )

        for i in range(cur_tp - 2):
            j, k = i + 1, i + 2

            w_ij = torch.abs(age_seq[j] - age_seq[i])
            w_jk = torch.abs(age_seq[k] - age_seq[j])

            weighted_avg = (w_ij * vol_roi[i] + w_jk * vol_roi[k]) / (w_ij + w_jk + epsilon)
            second_order_error = (vol_roi[j] - weighted_avg).pow(2)
            subj_loss.append(second_order_error)

        subj_loss_tensor = torch.stack(subj_loss)
        loss_list.append(subj_loss_tensor.mean())

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=volume.device, requires_grad=True) + epsilon
    else:
        return torch.stack(loss_list).mean() + epsilon

    
def smoothness_updated(get_volume, total_tp, ages, roi_val):

    #print(get_volume)
    #print(total_tp)
    #print(ages)

    ages = torch.tensor(ages, dtype=torch.float32, device='cuda')
    
    loss = []
    batch_temp = []
    for b in range(0, total_tp.shape[0], 1):
        
        cur_total_tp = total_tp[b] 
        #print("current total_tp", cur_total_tp)
        #print(get_volume[b])
        #print(ages[b])
        #print("i , j , k")

        whole_loss = []
        temp = []
        #print("Loop runs till 0 to ", cur_total_tp-2, "for time point of", total_tp[b] )
        for i in range(0, cur_total_tp-2, 1):
            #print(i)
            j = i+1
            k = i+2
            #print(i, j, k)
            if math.isnan(ages[b][i]):
                ages[b][i] = (ages[b][i-1] + ages[b][i+1])/2
                print("ages in i is nan", ages[b][i])
            if math.isnan(ages[b][j]):
                ages[b][j] = (ages[b][j-1] + ages[b][j+1])/2
                print("ages in j is nan", ages[b][j])
            if math.isnan(ages[b][k]):
                ages[b][k] = (ages[b][k-1] + ages[b][k+1])/2
                print("ages in k is nan", ages[b][k])
            #print("time point", i, "val", get_volume[b][i][roi_val], "age", ages[b][i])
            #print("time point", j, "val", get_volume[b][j][roi_val], "age", ages[b][j])
            #print("time point", k, "val", get_volume[b][k][roi_val], "age", ages[b][k])
            w_ij = torch.abs(ages[b][j] - ages[b][i])
            w_jk = torch.abs(ages[b][k] - ages[b][j])
            #print("w_ij", w_ij, "--- w_jk", w_jk)
            V_mean = ((w_ij * get_volume[b][i][roi_val]) + (w_jk * get_volume[b][k][roi_val])) / (w_ij + w_jk)
            #print(V_mean)
            #calc = nn.MSELoss(reduction='mean')
            #cur_loss = calc(get_volume[b][j] - V_mean)
            ### The expression torch.norm(a, p=2) ** 2 computes the squared Euclidean norm (or squared L2 norm) of the tensor a
            cur_loss = torch.norm(get_volume[b][j][roi_val] - V_mean, p=2) ** 2
            temp.append(cur_loss)
            whole_loss = torch.stack(temp)
            #print("######## whole losses", whole_loss)
            del V_mean
            #clear_gpu_memory()

        
        if (cur_total_tp>2):
            #print("mean of whole loss : ", torch.mean(whole_loss))
            batch_temp.append(torch.mean(whole_loss))
            loss = torch.stack(batch_temp)
        #print("for batch", b, "loss is:", loss)

    del whole_loss, temp, batch_temp
    clear_gpu_memory()

    if len(loss) == 0:
        loss = torch.tensor(0.0, requires_grad=True) + epsilon
    else:    
        loss = torch.mean(loss) + epsilon
        loss = loss + torch.tensor(0.0, requires_grad=True)
    return loss


def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def age_constraint_tensor(volume, total_tp, ages):
    
    # Move get_volume, total_tp, ages to GPU
    volume = torch.tensor(volume, dtype=torch.float32, device='cuda')
    total_tp = torch.tensor(total_tp, dtype=torch.int64, device='cuda')
    ages = torch.tensor(ages, dtype=torch.float32, device='cuda')
    
    #print("--- TENSOR FUNCTION")
    #print("volume type:", type(volume))
    #print("total_tp type:", type(total_tp))
    #print("ages type:", type(ages))

    B, T, R = volume.shape
    loss_list = []
    epsilon = 1e-8

    for b in range(B):
        cur_tp = int(total_tp[b].item())
        if cur_tp < 1:
            continue

        vol_LH = volume[b, :cur_tp, 1]
        vol_RH = volume[b, :cur_tp, 2]
        age_seq = ages[b, :cur_tp]

        # Handle missing ages using linear interpolation
        if torch.isnan(age_seq).any():
            valid_idx = (~torch.isnan(age_seq)).nonzero(as_tuple=True)[0]
            valid_vals = age_seq[valid_idx]
            interp_ages = torch.interp(
                torch.arange(cur_tp, device=volume.device, dtype=torch.float32),
                valid_idx.float(),
                valid_vals
            )
            age_seq = interp_ages

        # Compute expected volumes using the quadratic model
        actual_LH = quadratic_model(age_seq, 
                                    a=-0.18408087740848428, 
                                    b=12.972866265659233, 
                                    c=2737.2051019277146)
        actual_RH = quadratic_model(age_seq, 
                                    a=-0.4374767107074925, 
                                    b=49.34600659317594, 
                                    c=1546.8515880180278)

        # Compute absolute error between predicted and actual volumes
        error = torch.abs(actual_LH - vol_LH) + torch.abs(actual_RH - vol_RH)
        subj_loss = error.mean()
        loss_list.append(subj_loss)

    if len(loss_list) == 0:
        return torch.tensor(0.0, requires_grad=True, device=volume.device) + epsilon

    total_loss = torch.stack(loss_list).mean() / 1000.0 + epsilon
    return total_loss + torch.tensor(0.0, requires_grad=True, device=volume.device)

def age_constraint(get_volume, total_tp, ages):

    # Move get_volume, total_tp, ages to GPU
    #get_volume = torch.tensor(get_volume, dtype=torch.float32, device='cuda')
    #total_tp = torch.tensor(total_tp, dtype=torch.int64, device='cuda')
    #ages = torch.tensor(ages, dtype=torch.float32, device='cuda')

    #print("NON TENSOR FUNCTION")
    #print("get_volume type:", type(get_volume))
    #print("total_tp type:", type(total_tp))
    #print("ages type:", type(ages))

    #ages = torch.tensor(ages, dtype=torch.float32, device='cuda')
    
    loss = []
    batch_vol_total_np = []
    batch_vol_total = []
    #print("############ in age constraint ############")
    #print("Volumes", get_volume)
    #print("time point", total_tp)
    #print("ages", ages)
    
    for b in range(0, total_tp.shape[0], 1):

        #print("batch :", b)
        cur_total_tp = total_tp[b] 
        one_vol_total = []
        temp_full_error = []
        for ll in range(0, cur_total_tp, 1):
            cur_age = ages[b, ll]
            if (not math.isnan(cur_age)):
                my_vol_LH = get_volume[b, ll, 1]
                my_vol_RH = get_volume[b, ll, 2]
                #print("Age", cur_age.item(), "Volume LH", my_vol_LH.item(), "Volume RH", my_vol_RH.item())
                #actual_vol_LH = quadratic_model(x=cur_age, a=-0.18408087740848428, b=12.972866265659233, c=2737.2051019277146)
                #actual_vol_RH = quadratic_model(x=cur_age, a=-0.4374767107074925, b=49.34600659317594, c=1546.8515880180278)
                actual_vol_LH = torch.tensor(quadratic_model(x=cur_age, a=-0.18408087740848428, b=12.972866265659233, c=2737.2051019277146), dtype=torch.float32)
                actual_vol_RH = torch.tensor(quadratic_model(x=cur_age, a=-0.4374767107074925, b=49.34600659317594, c=1546.8515880180278), dtype=torch.float32)
                #print("Age", cur_age.item(), "actual LH", actual_vol_LH.item(), "actual RH", actual_vol_RH.item())
                error_LH = torch.abs(actual_vol_LH - my_vol_LH)
                error_RH = torch.abs(actual_vol_RH - my_vol_RH)
                #error_LH = np.abs(actual_vol_LH - my_vol_LH)
                #error_RH = np.abs(actual_vol_RH - my_vol_RH)
                full_error = error_LH + error_RH 
                temp_full_error.append(full_error)
                one_vol_total = torch.stack(temp_full_error)
            else:
                print("---------------------- nan found !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        del temp_full_error, full_error
            
        #print("before mean", one_vol_total)
        one_vol_total = torch.mean(one_vol_total)
        #print("after mean", one_vol_total)

        batch_vol_total_np.append(one_vol_total)
        batch_vol_total = torch.stack(batch_vol_total_np)
        #print("batch :", b, "total loss", batch_vol_total)

    #print("loss 3 batches", batch_vol_total)
    del one_vol_total, batch_vol_total_np
    clear_gpu_memory()
    
    loss = torch.mean(batch_vol_total)
    #print("loss3 val -", loss)
    loss = loss/1000
    loss = loss + epsilon
    loss = loss + torch.tensor(0.0, requires_grad=True)
    
    return loss 


def dice_score_variable_TP(preds, targets, time_lens, epsilon=1e-6):
    """
    Compute Dice score for variable timepoints in a batch.

    Args:
        preds: Tensor of shape (batch_size, T_max, H, W, D)
        targets: Tensor of same shape as preds
        time_lens: List of actual timepoints per subject (length = batch_size)
        epsilon: Small value to avoid division by zero

    Returns:
        mean_dice_score: scalar tensor representing average Dice score over time and batch
    """

    #print("\n####################################################################")
    #print("In dice score: Shape of pred is dice loss", preds.shape)
    #print("In dice score: Shape of targets is dice loss", targets.shape)
    #print("In dice score:TP", time_lens)

    #explore_3D_array_comparison(preds[0, 1, :, :, :].detach().cpu().numpy(), targets[0, 1, :, :, :].detach().cpu().numpy())
    
    batch_size = preds.shape[0]
    dice_score_batch = []

    for i in range(batch_size):
        T_i = time_lens[i]
        dice_score_time = []
        #print("\nHas total time point", T_i)
        for t in range(T_i):
            # Flatten the prediction and target at time t
            pred = preds[i, t].float().view(-1)  # Flatten the 3D volume to 1D
            target = targets[i, t].float().view(-1)

            intersection = (pred * target).sum()  # Intersection of prediction and target
            dice_score = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
            dice_score_time.append(dice_score)
            #print("Dice score at time", t, "is:", dice_score) 
 
        mean_dice_score_subject = torch.stack(dice_score_time).mean()  # Average Dice score for the subject
        #print("--------- Dice score mean", mean_dice_score_subject)
        dice_score_batch.append(mean_dice_score_subject)

    return torch.stack(dice_score_batch).mean()  # Average over all subjects in the batch


def dice_loss_variable_TP(preds, targets, time_lens, epsilon=1e-6):
    """
    Compute Dice loss for variable timepoints in a batch.

    Args:
        preds: Tensor of shape (batch_size, T_max, H, W, D)
        targets: Tensor of same shape as preds
        time_lens: List of actual timepoints per subject (length = batch_size)
        epsilon: Small value to avoid division by zero

    Returns:
        mean_dice_loss: scalar tensor representing average Dice loss over time and batch
    """
    
    #print("\n####################################################################")
    #print("In dice loss: Shape of pred is dice loss", preds.shape)
    #print("In dice loss: Shape of targets is dice loss", targets.shape)
    #print("In dice loss:TP", time_lens)

    
    batch_size = preds.shape[0]
    dice_loss_batch = []

    for i in range(batch_size):
        T_i = time_lens[i]
        dice_loss_time = []
        #print("Has total time point", T_i)
        for t in range(T_i):
            # Flatten the prediction and target at time t
            pred = preds[i, t].float().view(-1)  # Flatten the 3D volume to 1D
            target = targets[i, t].float().view(-1)

            intersection = (pred * target).sum()  # Intersection of prediction and target
            dice_score = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
            dice_loss = 1 - dice_score  # Dice loss is 1 - Dice score
            dice_loss_time.append(dice_loss)
            #print("Dice loss at time", t, "is:", dice_loss)

        mean_dice_loss_subject = torch.stack(dice_loss_time).mean()  # Average Dice loss for the subject
        dice_loss_batch.append(mean_dice_loss_subject)

    return torch.stack(dice_loss_batch).mean()  # Average over all subjects in the batch


def bce_loss_variable_TP(preds, targets, time_lens):
    """
    Compute Binary Cross Entropy loss for variable timepoints in a batch.

    Args:
        preds: Tensor of shape (batch_size, T_max, H, W, D) — logits
        targets: Tensor of same shape as preds (binary labels)
        time_lens: List of actual timepoints per subject (length = batch_size)

    Returns:
        mean_bce_loss: scalar tensor representing average BCE loss over time and batch
    """
    batch_size = preds.shape[0]  # Total number of subjects in the batch
    bce_loss_batch = []          # Will store average BCE loss per subject

    for i in range(batch_size):
        T_i = time_lens[i]       # Number of valid timepoints for this subject
        bce_loss_time = []       # Will store BCE loss at each timepoint for this subject

        for t in range(T_i):
            # Flatten the 3D volumes (H x W x D) at timepoint t into 1D tensors
            pred = preds[i, t].float().view(-1)    # Raw logits from the model
            target = targets[i, t].float().view(-1)  # Ground truth binary labels

            # Apply binary cross entropy loss with logits (internally uses sigmoid)
            loss = F.binary_cross_entropy_with_logits(pred, target)

            bce_loss_time.append(loss)  # Collect loss for this timepoint

        # Average BCE loss for this subject over all valid timepoints
        mean_bce_loss_subject = torch.stack(bce_loss_time).mean()
        bce_loss_batch.append(mean_bce_loss_subject)

    # Final loss: average BCE loss across all subjects in the batch
    return torch.stack(bce_loss_batch).mean()

import torch
import torch.nn as nn

def HighLevelFeatureConsistencyLoss(pred_features, target_features):
    """
    Compute MSE loss between predicted and target features across all timepoints.

    Args:
        pred_features: Tensor of shape [B, T, C, D, H, W]
        target_features: Tensor of same shape

    Returns:
        mse_loss: scalar tensor representing average MSE loss
    """
    mse_loss_fn = nn.MSELoss()
    return mse_loss_fn(pred_features, target_features)


def SCC_loss_TP(output_logits, pred_feat, targets, target_feat, total_tp):

    #print("Shape of all_pred-logits", output_logits.shape)
    #print("Shape of all_targets", targets.shape)

    #print("Shape of all_pred-after softmax", output.shape)
    #print("Uniques in preds", torch.unique(output))
    #print("Uniques in ground truth", torch.unique(targets))

    print("****** SSC Loss function: total time points in this sample", total_tp)
    print("Shape of feature target", pred_feat.shape)
    print("Shape of feature preds", target_feat.shape)
    
    num_classes = 3  # Adjust based on your actual number of classes
    
    #### getting the cross entropy loss
    ##########################################
    #the_bce_loss0 = BCELossTemporalLabelled((output_logits[:, :, 0, :, :, :]).float(), (targets[:, :, 0, :, :, :]).float())
    the_bce_loss0 = bce_loss_variable_TP(preds=output_logits[:, :, 0, :, :, :].float(), targets=targets[:, :, 0, :, :, :].float(), time_lens=total_tp)
    the_bce_loss1 = bce_loss_variable_TP(preds=output_logits[:, :, 1, :, :, :].float(), targets=targets[:, :, 1, :, :, :].float(), time_lens=total_tp)
    the_bce_loss2 = bce_loss_variable_TP(preds=output_logits[:, :, 2, :, :, :].float(), targets=targets[:, :, 2, :, :, :].float(), time_lens=total_tp)
    # Total loss is the sum of losses for all classes
    normalized_bce_loss = (0.1 * the_bce_loss0) + the_bce_loss1 + the_bce_loss2
    normalized_bce_loss = normalized_bce_loss / (0.1+1+1) 
    print("BCE loss", the_bce_loss0.item(), the_bce_loss1.item(), the_bce_loss2.item())

    output = F.softmax(output_logits, dim=2) # softmax should be done across the class dimension
    
    #### getting the dice scores
    ##############################################################################
    #dice2 = DiceScoreTemporalLabelled((output[:, :, 2, :, :, :]).float(), (targets[:, :, 2, :, :, :]).float())
    dice0 = dice_score_variable_TP(preds=output[:, :, 0, :, :, :].float(), targets=targets[:, :, 0, :, :, :].float(), time_lens=total_tp)
    dice1 = dice_score_variable_TP(preds=output[:, :, 1, :, :, :].float(), targets=targets[:, :, 1, :, :, :].float(), time_lens=total_tp)
    dice2 = dice_score_variable_TP(preds=output[:, :, 2, :, :, :].float(), targets=targets[:, :, 2, :, :, :].float(), time_lens=total_tp)

    #### getting the dice loss
    #########################################################################################
    #loss0 = DiceLossTemporalLabelled((output[:, :, 0, :, :, :]).float(), (targets[:, :, 0, :, :, :]).float())
    loss0 = dice_loss_variable_TP(preds=output[:, :, 0, :, :, :].float(), targets=targets[:, :, 0, :, :, :].float(), time_lens=total_tp)
    loss1 = dice_loss_variable_TP(preds=output[:, :, 1, :, :, :].float(), targets=targets[:, :, 1, :, :, :].float(), time_lens=total_tp)
    loss2 = dice_loss_variable_TP(preds=output[:, :, 2, :, :, :].float(), targets=targets[:, :, 2, :, :, :].float(), time_lens=total_tp)
    # Total loss is the sum of losses for all classes
    normalized_dice_loss = (0.1*loss0) + loss1 + loss2
    normalized_dice_loss = normalized_dice_loss / (0.1+1+1) # Normalize the total loss by the number of classes
    
    #### getting the high level feature consistency loss
    #########################################################################################
    hlfc =  HighLevelFeatureConsistencyLoss(pred_feat, target_feat)
    print("hlfc loss: ", hlfc.item())
    
    final_loss = normalized_dice_loss + normalized_bce_loss + (0.1*hlfc)
    #final_loss = normalized_dice_loss + hlfc
    #(0.45*normalized_dice_loss) + (0.45*normalized_bce_loss) + (0.1*hlfc)
    
    #fields=['dice', 'BCE', 'hlfc', 'loss1', 'smoothness loss', 'loss2', 'total_loss']
    all_of_them = [normalized_dice_loss.item(), normalized_bce_loss.item(), hlfc.item(), final_loss.item()]
    
    return final_loss, dice1, dice2, all_of_them



def differentiable_sdf_approx(mask: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """
    Differentiable approximation of Signed Distance Transform (SDF) for 3D binary masks.

    Args:
        mask (torch.Tensor): Tensor of shape [D, H, W] or [B, 1, D, H, W] with binary values (0 or 1).
        kernel_size (int): Size of the convolution kernel used to simulate distances.

    Returns:
        torch.Tensor: Differentiable signed distance transform (SDF) with same shape as input.
    """
    assert mask.ndim in (3, 5), "Expected shape [D, H, W] or [B, 1, D, H, W]"
    is_batched = (mask.ndim == 5)
    if not is_batched:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, D, H, W]

    device = mask.device
    K = kernel_size

    # Create coordinate grid
    coords = torch.stack(torch.meshgrid([
        torch.arange(K, device=device),
        torch.arange(K, device=device),
        torch.arange(K, device=device)
    ], indexing='ij'), dim=0)  # Shape: [3, K, K, K]

    center = (K - 1) / 2.0
    distances = torch.sqrt(((coords - center) ** 2).sum(0))  # [K, K, K]
    distances = distances / distances.max()  # normalize to [0, 1]
    kernel = distances.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, K, K, K]

    # Compute distances from foreground and background
    mask = mask.float()
    inv_mask = 1.0 - mask

    dist_inside = F.conv3d(mask, kernel, padding=K // 2)
    dist_outside = F.conv3d(inv_mask, kernel, padding=K // 2)

    sdf = dist_outside - dist_inside  # Positive outside, negative inside

    if not is_batched:
        sdf = sdf.squeeze(0).squeeze(0)  # Back to [D, H, W]
    return sdf



def compute_expansion_shrinkage(T1: torch.Tensor, T2: torch.Tensor) -> tuple:
    """
    Compute the weighted expansion and shrinkage between two masks using differentiable SDFs.
    Arguments:
        mask_t1: [D, H, W] Binary mask at time point 1 (e.g., initial time point)
        mask_t2: [D, H, W] Binary mask at time point 2 (e.g., later time point)
    Returns:
        expansion: Total weighted expansion
        shrinkage: Total weighted shrinkage
    """
    # Get the signed distance transform (SDF) for both time points using differentiable SDF approximation
    sdf_T1 = differentiable_sdf_approx(T1)
    sdf_T2 = differentiable_sdf_approx(T2)

    # Expansion: T2 foreground outside T1
    expansion_mask = (T1 == 0) & (T2 == 1)  # T2 foreground and positive sdf at T1 (outside boundary)
    #expansion_mask = (T2 == 1) & (sdf_T1 > 0)  # T2 foreground and positive sdf at T1 (outside boundary)
    expansion_weights = torch.zeros_like(sdf_T1)  # Initialize weights to zero
    expansion_weights[expansion_mask] = sdf_T1[expansion_mask]  # Assign sdf_T1 where the mask is True
    expansion_score = torch.sum(expansion_weights)#.to(torch.int)   # Sum the weighted expansion score
    
    # Shrinkage: T1 foreground not preserved in T2
    shrinkage_mask = (T1 == 1) & (T2 == 0)  # T1 foreground not present in T2 (shrinkage)
    #shrinkage_mask = (T1 == 1) & (sdf_T2 > 0)  # T1 foreground not present in T2 (shrinkage)
    shrinkage_weights = torch.zeros_like(sdf_T2)  # Initialize weights to zero
    shrinkage_weights[shrinkage_mask] = sdf_T2[shrinkage_mask]  # Assign sdf_T2 where the mask is True
    shrinkage_score = torch.sum(shrinkage_weights)#.to(torch.int)   # Sum the weighted shrinkage score

    #print(expansion_mask.shape, expansion_weights.shape)

    #print("the masks")
    #explore_3D_array_comparison(expansion_mask.cpu().numpy(), shrinkage_mask.cpu().numpy(), name1="Expansion", name2="Shrinkage")
    #print("the weights")
    #explore_3D_array_comparison(expansion_weights.cpu().numpy(), shrinkage_weights.cpu().numpy())

    #print("MAX MIN VALUES")
    #print("in expansion -- max", expansion_weights.max().item(), "min", expansion_weights.min().item())
    #print("in shrinkage -- max", shrinkage_weights.max().item(), "min", shrinkage_weights.min().item())
    #print("\nTotal expand", torch.sum(expansion_mask))
    #print("Total shrink", torch.sum(shrinkage_mask))
    #print("\nWeight sum expand", expansion_score)
    #print("Weight sum shrink", shrinkage_score)

    loss = torch.clamp(expansion_score - shrinkage_score, min=0)

    return loss


def SDF_loss_batch(output_logits, total_tps, label_no):

    predictions = F.softmax(output_logits, dim=2) # softmax should be done across the class dimension
    pred_class = torch.argmax(predictions, dim=2)  # shape [B, T, D, H, W] # Step 1: Get predicted class (argmax over channel dimension)
    binary_prediction = F.one_hot(pred_class, num_classes=3)  # shape [B, T, D, H, W, 3] # Step 2: Convert to one-hot encoding (binary mask)
    target = binary_prediction.permute(0, 1, 5, 2, 3, 4).float()  # Step 3: Move class dimension back to match original shape
    
    #print("*** Shape of all_pred-logits", output_logits.shape)
    #print("*** SDF Loss function: total time points in this sample", total_tps)
    #print("*** Shape of masks", target.shape)
    #print("*** Unique in masks", torch.unique(target))

    total_loss = []
    
    for cur_batch in range(0, total_tps.shape[0], 1):
        
        #print("*** Subject", cur_batch, "in batch has total tp", total_tps[cur_batch])
        cur_loss = []
        
        for cur_tp in range(1, total_tps[cur_batch], 1):
    
            #print("*** Vol shape", target[cur_batch, cur_tp, :, :, : ,:].shape)
            #explore_3D_array_comparison(target[cur_batch, cur_tp-1, label_no, :, : ,:].detach().cpu().numpy(), target[cur_batch, cur_tp, label_no, :, : ,:].detach().cpu().numpy(), name1="TP-"+str(cur_tp-1), name2="TP-"+str(cur_tp))
            loss_5 = compute_expansion_shrinkage(target[cur_batch, cur_tp-1, label_no, :, : ,:], target[cur_batch, cur_tp, label_no, :, : ,:])
            #print("*** LOSS 5:", loss_5)
    
            cur_loss.append(loss_5)
    
        cur_loss_avg = torch.stack(cur_loss).mean() 
        total_loss.append(cur_loss_avg)
        
        #print("\n\n-----This subject loss ---", cur_loss)
        #print("\n\n-----This subject loss avg---", cur_loss_avg)
        #print("\n\n-----Total loss ---", total_loss)
        #print("---------------------------------------\n\n")

    del cur_loss, cur_loss_avg
    clear_gpu_memory()
    
    #print("-------------------------------------------------------\n\n")
    total_loss_avg = torch.stack(total_loss).mean() 
    #print("-----Total loss avg---", total_loss_avg)
    #print("-------------------------------------------------------\n\n")

    return total_loss_avg

 
    

def get_other_demographics(id_list):

    all_ages_batch = []
    all_ICV_batch = []
    all_diagnosis_batch = []
    
    for cur_ID in id_list:
        
        # Filter rows where ID matches
        filtered_df = demographics[demographics['ID'].isin(cur_ID)]
        sorted_df = filtered_df.sort_values(by='Age', ascending=True)
        sorted_df["Diagnosis"] = sorted_df["Diagnosis"].replace(mapping).fillna(-1)
        
        #print(filtered_df)
        #print(sorted_df.shape)
        #print(sorted_df)
        #print("\n\n ------------ \n\n")
        
        all_ages = sorted_df['Age'].to_numpy()
        all_ICV = sorted_df['ICV'].to_numpy()
        all_diagnosis = sorted_df['Diagnosis'].to_numpy()

        for time in range(sorted_df.shape[0], max_tp, 1):
            #print("time value", time)
            all_ages = np.append(all_ages,-1)
            all_ICV = np.append(all_ICV, -1)
            all_diagnosis = np.append(all_diagnosis,-1)

        #print(all_ages)
        #print(all_ICV)
        #print(all_diagnosis)
        #print("\n ----------------------------------------------------- \n ----------------------------------------------------- \n")

        all_ages_batch.append(all_ages)
        all_ICV_batch.append(all_ICV)
        all_diagnosis_batch.append(all_diagnosis)

        #print(all_ages_batch.shape, all_ICV_batch.shape, all_diagnosis_batch.shape)  
        #print(len(all_ages_batch))  
        #print("all_ages_batch", all_ages_batch[0])  
        
    #print("Whole all ages", all_ages_batch)
    #print("\n\n ***************************************************** \n ***************************************************** \n\n")

    all_ages_batch = np.array(all_ages_batch)
    all_ICV_batch = np.array(all_ICV_batch)
    all_diagnosis_batch = np.array(all_diagnosis_batch)

    return all_ages_batch, all_ICV_batch, all_diagnosis_batch



class get_loss(nn.Module):

    def __init__(self):
        super(get_loss, self).__init__()
        
    def forward(self, output_logits, pred_feat, targets, target_feat, get_volume, total_tp, ages, all_IDs, cur_fold, loss_name):

        #target[target == 4] = 3 
        if (loss_name == "loss1"):
            print("In loss1 condition - DICE + BCE + HLFC")
            #print("prediction size", output_logits.size())
            #print("Shape of feature target", pred_feat.shape)
            #print("Shape of feature preds", target_feat.shape)
            loss_1, dice1, dice2, all_of_them = SCC_loss_TP(output_logits, pred_feat, targets, target_feat, total_tp)
            final_loss = loss_1
            #print("in loss1 type:", type(final_loss))
            #print("loss 1 ---", final_loss)
            del output_logits, pred_feat
            clear_gpu_memory()

        if(loss_name == "loss2"):
            print("In loss2 condition - SC")
            ages, _, _ = get_other_demographics(all_IDs)
            del _
            #print(get_volume)
            #get_volume = (get_volume/2097152)
            #get_volume = torch.tensor(get_volume, dtype=torch.float32).cuda()
            #loss_2 = smoothness(get_volume, total_tp)
            
            #loss_2_1 = smoothness_updated(get_volume, total_tp.numpy(), ages, 1)
            #loss_2_2 = smoothness_updated(get_volume, total_tp.numpy(), ages, 2)
            #final_loss = (loss_2_1 + loss_2_2) / 2
            
            loss_2_1_tensor = smoothness_updated_tensor(get_volume, total_tp.numpy(), ages, 1)
            loss_2_2_tensor = smoothness_updated_tensor(get_volume, total_tp.numpy(), ages, 2)
            final_loss = (loss_2_1_tensor + loss_2_2_tensor) / 2
            
            dice1 = dice2 = torch.tensor(0.0)
            #print("loss 2 ---", final_loss)
            #all_of_them = [-1, -1, -1, -1, loss_2.item(), final_loss.item()]
            all_of_them = final_loss.item()
            #del get_volume
            clear_gpu_memory()

        if(loss_name == "loss3"):
            print("In loss3 condition - AC")
            ages, _, _ = get_other_demographics(all_IDs)
            del _
            
            #loss_3 = age_constraint(get_volume, total_tp.numpy(), ages)
            loss_3 = age_constraint_tensor(get_volume, total_tp.numpy(), ages)
            
            #print("loss 3 ---", loss_3)
            #print("loss 3 (tensor) ---", loss_3_tensor)
            
            dice1 = dice2 = torch.tensor(0.0)
            final_loss = loss_3
            #final_loss = torch.tensor(0.0, requires_grad=True)
            
            all_of_them = final_loss.item()

        if(loss_name == "vol_loss"):

            print("In L4 - vol loss function")
            #print(all_IDs)
            ages, ICVs, diagnosis = get_other_demographics(all_IDs)
            diagnosis = np.array(diagnosis, dtype=np.int64)
            #print("DIAGNOSIS", diagnosis)
            #loss_4 = vol_loss_calculate(get_volume, total_tp.numpy(), ages, ICVs, diagnosis, cur_fold)
            #loss_4 = vol_loss_calculate_old(get_volume, total_tp.numpy(), ages, ICVs, diagnosis, cur_fold)
            #loss_4 = vol_loss_calculate_old_tensor(get_volume, total_tp.numpy(), ages, ICVs, diagnosis, cur_fold)
            loss_4 = torch.tensor(0.0, requires_grad=True)
            dice1 = dice2 = torch.tensor(0.0)
            final_loss = loss_4
            #print("loss 4 ---", final_loss)
            all_of_them = loss_4.item()

        if(loss_name == "sdf_loss"):
            print("In L5 - SDF loss function")
            #print("PREDS size", output_logits.shape)
            #print("TARGETS size", targets.shape)
            #loss_5_1 = SDF_loss_batch(output_logits, total_tp, label_no=1)
            #loss_5_2 = SDF_loss_batch(output_logits, total_tp, label_no=2)
            loss_5 = torch.tensor(0.0, requires_grad=True) 
            dice1 = dice2 = torch.tensor(0.0)
            #loss_5 = torch.stack([loss_5_1, loss_5_2]).mean()
            final_loss = loss_5
            all_of_them = loss_5.item()

        #print(all_of_them)
        #fields=['Dice', 'BCE', 'hlfc', 'loss1', 'Smoothness Loss OR loss2', 'Age Contraint Loss OR loss3', 'Mean Avg Volume loss', 'Total Loss']
        return final_loss, dice1.data, dice2.data, all_of_them

