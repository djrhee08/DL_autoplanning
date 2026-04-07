Preprocessing code for 2D MLC aperture stack to dose calculation project - For Yao Zhao

## Calculate dose from 2D MLC to dose directly, for single arc ##
- Input Data : 3D CT with electron density (not CT number) + treatment couch
               Stack of 2D MLC apertures from a single arc, the first array is from 181 gantry angle, the last array is from 179 gantry angle, with 2 deg gantry spacing (180 control points in total)

- Output Data : 3D Dose matched with 3D CT


## TO DO
- need to check if mlc/jaw stack in the same order, or we need to change it for gantry orientation each time.