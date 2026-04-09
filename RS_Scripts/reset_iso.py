# New iso should be near the target (+-3cm from the edge of the target)
#TODO : Make it as a class!
import itertools
import random

def dicts_to_key_lists(dict_list):
    # Initialize a dictionary to hold our lists
    result = {key: [] for key in dict_list[0].keys()}

    # Iterate over each dictionary
    for d in dict_list:
        # For each key, append the value to the corresponding list in the result
        for key in d:
            result[key].append(d[key])

    return result

def get_coordinate_limits(case, organ, CTName):
    tolerance = 0.5
    print("organ, CTName", organ, CTName)
    if case.PatientModel.StructureSets[CTName].RoiGeometries[organ].HasContours() == False:
        print("Empty organ, CTName", organ, CTName)
        return None
    
    roigeo = case.PatientModel.StructureSets[CTName].RoiGeometries[organ]
    if hasattr(roigeo.PrimaryShape, 'Contours') == False:
        roigeo.SetRepresentation(Representation="Contours")

    organ_contours = roigeo.PrimaryShape.Contours
    #coordinate_list = [item for sublist in organ_contours for item in sublsist]
    coordinate_list = list(itertools.chain(*organ_contours))
    
    coord_list = dicts_to_key_lists(coordinate_list)
    coord_list_x = coord_list['x']
    coord_list_y = coord_list['y']
    coord_list_z = coord_list['z']
    
    min_x = min(coord_list_x) - tolerance
    max_x = max(coord_list_x) + tolerance
    min_y = min(coord_list_y) - tolerance
    max_y = max(coord_list_y) + tolerance
    min_z = min(coord_list_z) - tolerance
    max_z = max(coord_list_z) + tolerance
    
    return [min_x, max_x, min_y, max_y, min_z, max_z]  

# find the min and max coordinates of targets
def find_coordinate_limit(case, CTName):
    coordinate_limits_list = []
    for organ in case.PatientModel.RegionsOfInterest:
        if organ.OrganData.OrganType.lower() == 'target':
            organName = organ.Name
            print("Target NAME: ", organName)
            coordinate_limits = get_coordinate_limits(case, organName, CTName)
            if coordinate_limits is not None:
                coordinate_limits_list.append(coordinate_limits)
            
    for idx, coordinate_limits in enumerate(coordinate_limits_list):
        if idx == 0:
            minmax_list = coordinate_limits
        else:
            if minmax_list[0] > coordinate_limits[0]:
                minmax_list[0] = coordinate_limits[0]
            if minmax_list[1] < coordinate_limits[1]:
                minmax_list[1] = coordinate_limits[1]
            if minmax_list[2] > coordinate_limits[2]:
                minmax_list[2] = coordinate_limits[2]
            if minmax_list[3] < coordinate_limits[3]:
                minmax_list[3] = coordinate_limits[3]
            if minmax_list[4] > coordinate_limits[4]:
                minmax_list[4] = coordinate_limits[4]
            if minmax_list[5] < coordinate_limits[5]:
                minmax_list[5] = coordinate_limits[5]                   
                                          
    return minmax_list

# define new iso position randomly within the coordinate limits of the target
def get_newiso(case, CTName, beam_set):
    minmax_list = find_coordinate_limit(case, CTName)
    current_iso = beam_set.Beams[0].Isocenter.Position

    x_min = max(current_iso['x'] - 3, minmax_list[0])
    x_max = min(current_iso['x'] + 3, minmax_list[1])
    x_iso = round(random.uniform(x_min, x_max), 2)

    y_min = max(current_iso['y'] - 3, minmax_list[2])
    y_max = min(current_iso['y'] + 3, minmax_list[3])
    y_iso = round(random.uniform(y_min, y_max), 2)

    z_min = max(current_iso['z'] - 3, minmax_list[4])
    z_max = min(current_iso['z'] + 3, minmax_list[5])
    z_iso = round(random.uniform(z_min, z_max), 2)

    return x_iso, y_iso, z_iso

# reset isocenter position
def reset_isocenter(case, beam_set, CTName, IsoName='zz_test_iso'):
    x_iso, y_iso, z_iso = get_newiso(case, CTName, beam_set)
    
    # Apply new iso to the beam
    for beam in beam_set.Beams:
        beam.Isocenter.EditIsocenter(Name=IsoName, Color="98, 184, 234", Position={ 'x': x_iso, 'y': y_iso, 'z': z_iso })