import matplotlib.pyplot as plt

from ipywidgets import interact
import numpy as np
import SimpleITK as sitk
import cv2

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  def fn(SLICE):
    plt.figure(figsize=(4,4))
    plt.imshow(arr[SLICE, :, :], cmap=cmap)

  interact(fn, SLICE=(0, arr.shape[0]-1))


def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, name1: str = "Prediction", name2: str = "Ground Truth", cmap: str = 'gray'):
  """
  Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
  The purpose of this function to visual compare the 2D arrays after some transformation. 

  Args:
    arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
    arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  assert arr_after.shape == arr_before.shape

  def fn(SLICE):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(7,7))

    ax1.set_title(name1, fontsize=15)
    ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

    ax2.set_title(name2, fontsize=15)
    ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

    plt.tight_layout()
  
  interact(fn, SLICE=(0, arr_before.shape[0]-1))
    

def explore_3D_array_comparison_three(arr_1: np.ndarray, arr_2: np.ndarray, arr_3: np.ndarray, cmap: str = 'gray'):

    #assert arr_1.shape == arr_2.shape
  
    def fn(SLICE):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(7,14))

        ax1.set_title('Volume', fontsize=15)
        ax1.imshow(arr_1[SLICE, :, :], cmap=cmap)
        
        ax2.set_title('Ground Truth', fontsize=15)
        ax2.imshow(arr_2[SLICE, :, :], cmap=cmap)

        ax3.set_title('Prediction', fontsize=15)
        ax3.imshow(arr_3[SLICE, :, :], cmap=cmap)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_1.shape[0]-1))

def explore_3D_array_overlay_GT(arr_3: np.ndarray, arr_1: np.ndarray, arr_2: np.ndarray, my_title: str, cmap: str = 'gray'):

    #assert arr_1.shape == arr_2.shape

    def fn(SLICE):
        fig, (ax4, ax1, ax2, ax3) = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(7,14))

        ax4.set_title(my_title, fontsize=15)
        ax4.imshow(arr_3[SLICE, :, :], cmap=cmap)
        
        ax1.set_title('Ground Truth', fontsize=15)
        ax1.imshow(arr_1[SLICE, :, :], cmap=cmap)
        
        ax2.set_title('Prediction', fontsize=15)
        ax2.imshow(arr_2[SLICE, :, :], cmap=cmap)

        ax3.set_title('Pred on GT', fontsize=15)
        ax3.imshow(arr_1[SLICE, :, :], cmap='gray')
        ax3.imshow(arr_2[SLICE, :, :], cmap='jet', alpha=0.3)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_1.shape[0]-1))


def explore_3D_array_overlay_VOL(arr_1: np.ndarray, arr_2: np.ndarray, cmap: str = 'gray'):

    #assert arr_1.shape == arr_2.shape

    def fn(SLICE):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(7,14))

        ax1.set_title("Volume", fontsize=15)
        ax1.imshow(arr_1[SLICE, :, :], cmap=cmap)
        
        ax2.set_title('Ground Truth', fontsize=15)
        ax2.imshow(arr_2[SLICE, :, :], cmap=cmap)
        
        ax3.set_title('Pred on GT', fontsize=15)
        ax3.imshow(arr_1[SLICE, :, :], cmap='gray')
        ax3.imshow(arr_2[SLICE, :, :], cmap='jet', alpha=0.3)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_1.shape[0]-1))


def explore_3D_array_registration(arr_1: np.ndarray, arr_2: np.ndarray, arr_3: np.ndarray, cmap: str = 'gray'):

    #assert arr_1.shape == arr_2.shape
  
    def fn(SLICE):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(7,14))

        ax1.set_title('Fixed Image', fontsize=15)
        ax1.imshow(arr_1[SLICE, :, :], cmap=cmap)
        
        ax2.set_title('Registered Image on Fixed Image', fontsize=15)
        ax2.imshow(arr_1[SLICE, :, :], cmap='gray')
        ax2.imshow(arr_2[SLICE, :, :], cmap='jet', alpha=0.3)

        ax3.set_title('Moving Image', fontsize=15)
        ax3.imshow(arr_3[SLICE, :, :], cmap=cmap)

        ax4.set_title('Moving Image on Fixed Image', fontsize=15)
        ax4.imshow(arr_1[SLICE, :, :], cmap='gray')
        ax4.imshow(arr_3[SLICE, :, :], cmap='jet', alpha=0.3)

        #ax4.set_title('Overlay Image', fontsize=15)
        #ax4.imshow(arr_1[SLICE, :, :], cmap='gray')
        #ax4.imshow(arr_2[SLICE, :, :], cmap='jet', alpha=0.5)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_1.shape[0]-1))


def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')


def add_suffix_to_filename(filename: str, suffix:str) -> str:
  """
  Takes a NIfTI filename and appends a suffix.

  Args:
      filename : NIfTI filename
      suffix : suffix to append

  Returns:
      str : filename after append the suffix
  """
  if filename.endswith('.nii'):
      result = filename.replace('.nii', f'_{suffix}.nii')
      return result
  elif filename.endswith('.nii.gz'):
      result = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
      return result
  else:
      raise RuntimeError('filename with unknown extension')


def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b


def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
  mask provided will be used to overlay contours of the region of interest over the 
  array. The purpose of this function is to visual inspect the region delimited by the mask.

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    mask : binary mask to obtain the region of interest
  """
  assert arr.shape == mask.shape
  
  _arr = rescale_linear(arr,0,1)
  _mask = rescale_linear(mask,0,1)
  _mask = _mask.astype(np.uint8)

  def fn(SLICE):
    arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), thickness)

    plt.figure(figsize=(7,7))
    plt.imshow(arr_with_contours)

  interact(fn, SLICE=(0, arr.shape[0]-1))
