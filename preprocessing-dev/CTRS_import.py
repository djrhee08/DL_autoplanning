import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import pathlib
import sys
import os
import glob
from matplotlib.path import Path

class DICOMCTRS_importer():
    def __init__(self):
        print("DICOMCTRS importer initialized")
            
    def read_dicom_CT(self, CT_dir, CTtoED_curve=None):
        current_path = os.path.join(CT_dir, "*.dcm")
        CT_idx = 0
        rts_idx = 0
        CT_list = []
        
        for dcm in glob.glob(current_path):
            ds = dicom.dcmread(dcm)
            if ds.Modality == 'RTSTRUCT':
                rts_idx += 1
                if rts_idx != 1:
                    print("There are more than 1 RTStructure file in the directory!")
                    sys.exit()
                contour_data, edensity_data = self.get_structures(ds)
            
            elif ds.Modality == 'RTDOSE':
                pass
            
            elif ds.Modality == 'RTPLAN':
                pass
                    
            elif ds.Modality == 'CT':
                if CT_idx == 0:
                    if hasattr(ds, 'SpacingBetweenSlices'):
                        SliceThickness = ds.SpacingBetweenSlices
                    else:
                        SliceThickness = ds.SliceThickness
                    PixelSpacing = ds.PixelSpacing
                    RescaleSlope = ds.RescaleSlope
                    RescaleIntercept = ds.RescaleIntercept
                    PatientPosition = ds.PatientPosition
                    
                    # TODO: See if the code works for non-HFS orientations
                    if PatientPosition != "HFS":
                        print("PatientPosition is not HFS, but", PatientPosition, "! Please check it to see if it works")
                        sys.exit()
                    ImagePositionPatient = ds.ImagePositionPatient
                    CT_idx += 1
                else:
                    if hasattr(ds, 'SpacingBetweenSlices'):
                        temp_SliceThickness = ds.SpacingBetweenSlices
                    else:
                        temp_SliceThickness = ds.SliceThickness
                    if temp_SliceThickness != SliceThickness:
                        print("The slice thickness is not consistent", SliceThickness, temp_SliceThickness)
                        sys.exit()
                    if ds.PixelSpacing[0] != PixelSpacing[0]:
                        print("The x pixel spacing is not consistent", PixelSpacing[0], ds.PixelSpacing[0])
                        sys.exit()
                    if ds.PixelSpacing[1] != PixelSpacing[1]:
                        print("The y pixel spacing is not consistent", PixelSpacing[1], ds.PixelSpacing[1])
                        sys.exit()
                        
                CT_list.append(dcm)          
            
        # Create 3D numpy array from CT DICOM files
        # As numpy to SimpleITK conversion is done, the order of the axis is reversed (i.e. (z, y, x) -> (x, y, z))
        if 'PixelSpacing' not in locals():
            raise ValueError("PixelSpacing is not defined in the DICOM files.")
        
        pixel_spacing = [PixelSpacing[0], PixelSpacing[1], SliceThickness] # (x, y, z)
        slope_intercept = [RescaleSlope, RescaleIntercept]
        CT_list.sort()

        CT_sitk, CT_info = self.convert_dicom_to_sitk(CT_list, ImagePositionPatient, pixel_spacing, slope_intercept, CTtoED_curve)
        masks_sitk = self.contour_to_binary_mask(contour_data, CT_info)

        # Apply masks in CT images to have final CT images
        CT_sitk = self.merge_masks_to_CT(CT_sitk, masks_sitk, edensity_data)

        #### THIS IS FOR DEBUGGING ####
        """
        CT_array = sitk.GetArrayFromImage(CT_sitk) 
        np.save(os.path.join("./CT_array.npy"), CT_array)

        mask_external = masks_sitk['EXTERNAL'][0]
        np.save(os.path.join("./external_array.npy"), mask_external)
        
        mask_couch = masks_sitk['VMS IGRTCT 6MV d'][0]
        np.save(os.path.join("./couch_array.npy"), mask_couch)
        """
        #######################################################

        return CT_sitk
    

    def convert_dicom_to_sitk(self, img_list, ImagePositionPatient, pixel_spacing, slope_intercept, CTtoED_curve):
        """
        convert dicom to sitk, with electron density voxel values

        Returns:
            sitk_image (bool): CT image in sitk format, with electron density voxel value. (x, y, z)
        """
        # Read the DICOM files
        dicom_slices = []
        min_slice = dicom.dcmread(img_list[0]).ImagePositionPatient[2]
        
        for file in img_list:
            ds = dicom.dcmread(file)
            dicom_slices.append(ds)
            current_slice = ds.ImagePositionPatient[2]
            min_slice = min(min_slice, current_slice)
        
        # Need to use either ImagePositionPatient, not SliceLocation, although they are supposed to be the same
        # no need to flip the image in z axis anymore, so reverse=False
        dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=False)
        
        sorted_slice_positions = [ds.ImagePositionPatient[2] for ds in dicom_slices]
        slice_spacings = np.array([j - i for i, j in zip(sorted_slice_positions[:-1], sorted_slice_positions[1:])])
        
        if np.any(np.abs(slice_spacings - slice_spacings[0]) > 0.01):
            print(slice_spacings)
            print("Slice spacing are inconsistent!!! check the dicom image files")
            sys.exit()
        
        # For 4D CT, SliceThickness can be None
        if pixel_spacing[2] is None:
            pixel_spacing[2] = np.abs(slice_spacings[0])
            print("SliceThickness not available. Probably a 4D CT. pixel_spacing is updated with slice spacing", pixel_spacing[2])
        else:    
            if np.abs(slice_spacings[0] - pixel_spacing[2]) > 0.01:
                print("slicespacing and pixel_spacing ", slice_spacings[0], pixel_spacing[2], " are inconsistent!")
                pixel_spacing[2] = np.abs(slice_spacings[0])
                print("pixel_spacing is updated to ", pixel_spacing[2])

        # Extract pixel array from DICOM slices
        dicom_array = np.stack([slice.pixel_array for slice in dicom_slices]) # (z, x, y)
        
        # Apply the gradient and intercept to obtain CT numbers
        dicom_array = dicom_array * slope_intercept[0] + slope_intercept[1]
        
        # Convert CT number to electron density
        dicom_array = self.apply_edensity_curve(dicom_array, CTtoED_curve) # This is optional for MDA cases, but mandatory for other institutions

        # no transpose, but keep the original order
        sitk_image = sitk.GetImageFromArray(dicom_array)
        
        # Set pixel spacing
        sitk_image.SetSpacing(pixel_spacing)

        # Calculate the image origin based on the ImagePositionPatient
        # As numpy to SimpleITK conversion is done, the order of the axis is reversed (i.e. (z, y, x) -> (x, y, z))
        new_origin = [ImagePositionPatient[0], ImagePositionPatient[1], min_slice] #(x, y, z)

        sitk_image.SetOrigin(new_origin)

        CT_info = {} # (x, y, z), as the contour coordinates are saved in this order
        CT_info['image_shape'] = [dicom_array.shape[1], dicom_array.shape[2], dicom_array.shape[0]]
        CT_info['spacing'] = [pixel_spacing[0], pixel_spacing[1], pixel_spacing[2]]
        CT_info['origin'] = [ImagePositionPatient[0], ImagePositionPatient[1], min_slice]
        
        return sitk_image, CT_info

    def get_structures(self, ds):
        """
        This function extract External to remove anything outside External 0,
                              couch to override density
                              special structures to override density (e.g. skin collimation)
                              and any overriden structures to override density.
        """
        # Get contour info from the useful contours
        contour_data = {}
        roi_data = {}
        edensity_data = {}
        roi_name_list = []
        
        for ob_sequence in ds.RTROIObservationsSequence:
            # Record External contour
            if ob_sequence.RTROIInterpretedType.upper() in ['EXTERNAL']:
                roi_number = ob_sequence.ReferencedROINumber
                roi_type = ob_sequence.RTROIInterpretedType.upper()
                roi_data[roi_number] = roi_type
                roi_name_list.append(roi_type)
                edensity_data[roi_type] = 0
            # Record any other structures that need to be overriden (e.g. couch, skin collimation)
            if hasattr(ob_sequence, 'ROIPhysicalPropertiesSequence'):
                for physicalproperty in ob_sequence.ROIPhysicalPropertiesSequence:
                    if physicalproperty.ROIPhysicalProperty == 'REL_ELEC_DENSITY':                       
                        roi_number = ob_sequence.ReferencedROINumber
                        roi_name = ob_sequence.ROIObservationLabel
                        roi_data[roi_number] = roi_name
                        roi_name_list.append(roi_name)
                        edensity_data[roi_name] = physicalproperty.ROIPhysicalPropertyValue
                        
        for contour_sequence in ds.ROIContourSequence:
            roi_number = contour_sequence.ReferencedROINumber
            roi_name = roi_data.get(roi_number)
            if roi_name in roi_name_list:
                contours = []
                contour_data[roi_name] = []

                for contour_sequence_item in contour_sequence.ContourSequence:
                    contour_slice = np.array(contour_sequence_item.ContourData).reshape(-1, 3)
                    contours.append(contour_slice[:, :3])  # Keep x, y, z coordinates

                contour_data[roi_name] = contours

        return contour_data, edensity_data


    def merge_masks_to_CT(self, CT_sitk, masks_sitk, edensity_data):
        """
        Merge binary masks with the CT image to create a final CT image with masks applied.
        
        Parameters:
            CT_sitk (SimpleITK.Image): The original CT image in SimpleITK format.
            masks_sitk (dict): Dictionary of binary masks in SimpleITK format.
        
        Returns:
            merged_CT (SimpleITK.Image): The CT image with masks applied.
        """
        # Convert CT image to numpy array
        CT_array = sitk.GetArrayFromImage(CT_sitk)
        
        # electron density outside the external contour becomes 0
        is_external = False
        for contour_type, mask_array in masks_sitk.items():
            if contour_type == "EXTERNAL":
                is_external = True
                CT_array[mask_array[0] == 0] = 0  # Example: Set outside masked regions to 0
                #np.save('external_mask.npy', mask_array)
        
        if is_external == False:
            print("EXTERNAL CONTOUR DOES NOT EXIST!!")
            sys.exit()
            
        # Another for loop for overriden structures - not in the same for loop as the order matters!
        # e.g. override couch -> override external would delete the couch density
        for contour_type, mask_array in masks_sitk.items():
            if contour_type != "EXTERNAL":
                edensity = edensity_data[contour_type]  # Override density
                CT_array[mask_array[0] == 1] = edensity
        
        # Convert the modified numpy array back to SimpleITK image
        merged_CT = sitk.GetImageFromArray(CT_array)
        merged_CT.CopyInformation(CT_sitk)
        
        return merged_CT


    def contour_to_binary_mask(self, contour_data, CT_info):
        """
        Convert contour coordinates into a full 3D binary mask by drawing each contour
        on its representative slice. The even–odd rule is applied per drawn slice to preserve donut shapes in the x–y plane.
        The final mask orientation is [z, x, y] to match CT images.

        Parameters:
            contour_data (dict): Dictionary of contour coordinates.
                                Format: {
                                    contour_type: [
                                        [
                                            (x1, y1, z1), (x2, y2, z2), ...
                                        ],
                                        ...
                                    ],
                                    ...
                                }
            CT_info (dict): Dictionary containing CT image information with keys:
                            'shape' (tuple): The shape of the CT image in [z, x, y].
                            'spacing' (tuple): The spacing in [z, x, y].
                            'origin' (tuple): The origin in DICOM coordinates (x0, y0, z0).

        Returns:
            mask_total (dict): Dictionary of binary masks for each contour type.
                            Each mask is a 3D numpy array in [z, x, y].
        """
        # Extract CT image parameters
        # CT image shape: [x, y, z]
        x_dim, y_dim, z_dim = CT_info['image_shape']
        x_spacing, y_spacing, z_spacing = CT_info['spacing']
        origin_x, origin_y, origin_z = CT_info['origin']

        # Prepare output dictionary
        mask_total = {contour_type: [] for contour_type in contour_data.keys()}

        # For each contour type, create a 3D mask
        for contour_type, contours_list in contour_data.items():
            # Create empty 3D mask (z, x, y)
            contour_mask_3d = np.zeros((z_dim, x_dim, y_dim), dtype=np.uint8)

            # Process each contour in this contour type
            for contour in contours_list:
                # Convert each (x, y, z) to slice index and pixel indices
                # We'll assume the CT slices are oriented in the z direction.
                # 1) Convert DICOM world coordinates to slice index (z index)
                # 2) Convert x, y to pixel coordinates
                #    pixel_x = (world_x - origin_x) / x_spacing
                #    pixel_y = (world_y - origin_y) / y_spacing
                # 3) z index = round((world_z - origin_z) / z_spacing)

                # Separate out coordinates for clarity
                x_coords_world = np.array([pt[0] for pt in contour])
                y_coords_world = np.array([pt[1] for pt in contour])
                z_coords_world = np.array([pt[2] for pt in contour])

                # Compute slice indices (assuming all points in the same contour belong to one slice)
                z_indices_float = (z_coords_world - origin_z) / z_spacing
                z_index_approx = np.round(z_indices_float).astype(int)

                # If the contour points lie in multiple slices (which can happen with multi-slice contours),
                # they can be handled slice-by-slice or with a more advanced interpolation.
                # Below assumes they all belong to the same slice, taking the median.
                z_index = int(np.median(z_index_approx))

                # Verify slice is within bounds
                if z_index < 0 or z_index >= z_dim:
                    continue

                # Convert x, y to pixel indices
                x_indices_float = (x_coords_world - origin_x) / x_spacing
                y_indices_float = (y_coords_world - origin_y) / y_spacing

                # Build a polygon path for the even–odd fill rule
                polygon_pts = np.vstack((x_indices_float, y_indices_float)).T
                path = Path(polygon_pts, closed=True)

                # Create a grid of points for the slice
                x_grid, y_grid = np.mgrid[0:x_dim, 0:y_dim]
                # Flatten and combine into Nx2 for Path.contains_points()
                points = np.vstack((x_grid.ravel(), y_grid.ravel())).T

                # Use the even–odd rule to fill the polygon
                # points_inside is a boolean array of shape (x_dim*y_dim,)
                points_inside = path.contains_points(points, radius=1e-9)
                points_inside = points_inside.reshape((x_dim, y_dim))

                # Combine with the existing 2D slice mask using logical OR
                contour_mask_3d[z_index, :, :] ^= points_inside.astype(np.uint8)

            # After processing all contours in this type, save to output dictionary
            # Transpose back to original [x, y, z] orientation.
            contour_mask_3d = contour_mask_3d.transpose(0, 2, 1)
            mask_total[contour_type].append(contour_mask_3d)

        return mask_total


    def apply_edensity_curve(self, dicom_array, CTtoED_curve):
        """
        This function applies the CTnumber to the electron density curve to the CT image
        """
        if CTtoED_curve == None:
            current_dir = pathlib.Path(__file__).parent.resolve()
            CTtoED_curve = os.path.join(current_dir, "CTtoED.txt")
            
        ED = []
        CT = []
        with open(CTtoED_curve, 'r') as f:
            for line in f:
                CT.append(float(line.split('\t')[0].rstrip()))
                ED.append(float(line.split('\t')[1].rstrip()))
        
        dicom_array = np.interp(dicom_array, np.array(CT), np.array(ED))
        
        return dicom_array