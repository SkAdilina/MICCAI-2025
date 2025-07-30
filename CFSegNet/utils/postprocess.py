import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
import torch

def biasCorrection(sitk_img):

    print("bias correcting ...")

    # Extracting the head mask as only want to process the region where the head is for the bias correction.
    transformed = sitk.RescaleIntensity(sitk_img, 0, 255)
    transformed = sitk.LiThreshold(transformed,0,1)
    head_mask = transformed
    
    # the visualizing the sitk image with its mask
    #explore_3D_array_comparison(arr_before=sitk.GetArrayFromImage(sitk_img), arr_after=sitk.GetArrayFromImage(head_mask))
    
    # shrinking the image becuase it will take a long time to process
    shrinkFactor = 4
    inputImage = sitk_img
    #inputImage = sitk.Cast(sitk_img, sitk.sitkFloat64)

    inputImage = sitk.Shrink( sitk_img, [ shrinkFactor ] * inputImage.GetDimension() )
    maskImage = sitk.Shrink( head_mask, [ shrinkFactor ] * inputImage.GetDimension() )

    # defining and performing the bias correction
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)
    
    # bringing back the image to its original resolution
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(sitk_img)
    log_bias_field = sitk.Cast(log_bias_field, sitk.sitkFloat64)
    normalized_img = sitk_img / sitk.Exp( log_bias_field )
    
    """
    # to better visualizing the bias field
    temp = sitk.Exp(log_bias_field)
    temp = sitk.Mask(temp, head_mask)
    explore_3D_array(sitk.GetArrayFromImage(temp), cmap='gray')
    """
    
    # looking at the original and the bias corrected image side by side
    #explore_3D_array_comparison(sitk.GetArrayFromImage(sitk_img), sitk.GetArrayFromImage(normalized_img), cmap='nipy_spectral')
    
    return normalized_img

def fullZScoreNormalization(sitk_img):

    """
    ### Z-score normalization
    
    """
    print("Z-score normalizing ....")
    
    img_array = sitk.GetArrayFromImage(sitk_img)
    #print(f"Image array dtype: {img_array.dtype}")
   
    # Calculate mean and standard deviation
    mean = np.mean(img_array)
    std = np.std(img_array)
    #print("before : ", mean, std)
    
    # Perform Z-score normalization
    normalized_img_array = (img_array - mean) / std
    #print("after : ", np.mean(normalized_img_array), np.std(normalized_img_array))
    #print('\n')

    # Convert numpy array back to SimpleITK image
    normalized_img = sitk.GetImageFromArray(normalized_img_array)
    normalized_img.CopyInformation(sitk_img)  # Copy meta-information from original image
    
    return normalized_img


def intensity_normalize(sitk_img):
    
    normalized_img = biasCorrection(sitk_img)
    normalized_img = fullZScoreNormalization(normalized_img)
    
    # looking at the original and the bias corrected image side by side
    #explore_3D_array_comparison(sitk.GetArrayFromImage(sitk_img), sitk.GetArrayFromImage(normalized_img), cmap='nipy_spectral')
    
    return normalized_img

def intensity_normalize_for_nnunet_preprocess(sitk_img):

    normalized_img = fullZScoreNormalization(sitk_img)
    
    return normalized_img

def cropping_function(volume, vol_crop_shape):

    #print("in cropping_function", volume.shape, vol_crop_shape)
    if np.squeeze(volume).shape == vol_crop_shape:
        print("***** No cropping is needed!")
        return volume

    crop_size = vol_crop_shape
    image = volume
    
    #print("image shape in crop", image.shape)
    _, depth, height, width = image.shape
  
    # Calculate the starting indices for cropping
    start_depth = (depth - crop_size[0]) // 2
    start_height = (height - crop_size[1]) // 2
    start_width = (width - crop_size[2]) // 2

    # Calculate the ending indices for cropping
    end_depth = start_depth + crop_size[0]
    end_height = start_height + crop_size[1]
    end_width = start_width + crop_size[2]

    # Crop the volume
    image = image[:, start_depth:end_depth, start_height:end_height, start_width:end_width]

    return image


def reverse_cropping_function(cropped_volume, original_shape):

    original_shape = (1, ) + original_shape

    if cropped_volume.shape == original_shape:
        print("***** No reverse cropping needed!")
        return cropped_volume
    
    _, original_depth, original_height, original_width = original_shape
    _, cropped_depth, cropped_height, cropped_width = cropped_volume.shape

    print("The shapes during padding - original:", original_shape, "cropped:", cropped_volume.shape)
    
    # Calculate the starting indices for padding
    start_depth = (original_depth - cropped_depth) // 2
    start_height = (original_height - cropped_height) // 2
    start_width = (original_width - cropped_width) // 2
    
    # Create an empty tensor with the original shape
    restored_volume = torch.zeros(original_shape, dtype=cropped_volume.dtype, device=cropped_volume.device)
    
    # Place the cropped volume back into the center of the empty tensor
    restored_volume[:, 
                    start_depth:start_depth+cropped_depth, 
                    start_height:start_height+cropped_height, 
                    start_width:start_width+cropped_width] = cropped_volume
    
    return restored_volume


"""
def pad_to_original(cropped_image, original_shape):

    if cropped_image.shape == original_shape:
        print("No padding needed!")
        return cropped_image
    
    # Original dimensions
    original_depth, original_height, original_width = original_shape
    
    # Cropped dimensions
    _, cropped_depth, cropped_height, cropped_width = cropped_image.shape

    print("the shapes during padding - original -", original_shape, "cropped -", cropped_image.shape)
    
    # Calculate the padding amounts
    pad_depth = (original_depth - cropped_depth) // 2
    pad_height = (original_height - cropped_height) // 2
    pad_width = (original_width - cropped_width) // 2
    
    # Handle cases where the difference is odd
    pad_depth_extra = (original_depth - cropped_depth) % 2
    pad_height_extra = (original_height - cropped_height) % 2
    pad_width_extra = (original_width - cropped_width) % 2
    
    # Padding: (left, right, top, bottom, front, back)
    padding = (pad_width, pad_width + pad_width_extra, 
               pad_height, pad_height + pad_height_extra, 
               pad_depth, pad_depth + pad_depth_extra)
    
    # Apply padding
    padded_image = F.pad(cropped_image, padding)
    
    return padded_image

"""


