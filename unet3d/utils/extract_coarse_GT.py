import numpy as np
import nibabel as nib

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    new_header = header=nifti_image.header.copy()
    image_affine = nifti_image.affine
    output_nifti = nib.nifti1.Nifti1Image(image_numpy, None, header=new_header)
    nib.save(output_nifti, output_path)

def extract_3D_bbox(img):
    # Source: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return cmin, rmin, zmin, cmax, rmax, zmax

def crop_to_bbox_3D(image, bbox, preserve_tumor_shape_force_square_crop = False, crop_margin=2):
    """
    Crop an image to the bounding by forcing a squared image as output.
    """
    x1, y1, z1, x2, y2, z2 =  bbox

    if preserve_tumor_shape_force_square_crop == True:

        max_width_height = max(y2 - y1, x2 - x1, z2 - z1)
        # force a squared image in x and y direction
        y2 = y1 + max_width_height
        x2 = x1 + max_width_height
        # z2 = z1 + max_width_height

    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0)
    y2 = np.minimum(y2 + crop_margin, image.shape[0])
    x1 = np.maximum(x1 - crop_margin, 0)
    x2 = np.minimum(x2 + crop_margin, image.shape[1])
    z1 = np.maximum(z1 - crop_margin, 0)
    z2 = np.minimum(z2 + crop_margin, image.shape[2])

    cropped_image = image[y1:y2, x1:x2, z1:z2]


    foreground_mask = np.zeros_like(image)
    foreground_mask[y1:y2, x1:x2, z1:z2] = 1    
    masked_image = image * foreground_mask
    image_bg = image[0,0,0]
    masked_image[masked_image == 0] = image_bg

    return cropped_image, masked_image

