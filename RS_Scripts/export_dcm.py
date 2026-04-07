# From RayStation 2025, connect module is replaced by rayastation module.
try:
    # Standard RayStation scripting environment
    from connect import *
    print("Using RayStation connect module")
except ImportError:
    # Alternative environment (e.g. research / RayStation Python API wrapper)
    from raystation import *
    print("Using raystation module")
import os
import sys
import numpy as np

# Get fluence data and convert it into 2D
def get_fluence_map(FluenceData, NrBixels, Corner, BixelWidth):
    if NrBixels.x * NrBixels.y != len(FluenceData):
        print("Number of pixels are different from the actual number of data points")
        sys.exit()
    
    cntr_x = Corner.x + BixelWidth * (NrBixels.x / 2)
    cntr_y = Corner.x + BixelWidth * (NrBixels.x / 2)
    
    if cntr_x > abs(BixelWidth/2):
        print("Center (x) of the fluence map is not at the isocenter", cntr_x)
        sys.exit()
    if cntr_y > abs(BixelWidth/2):
        print("Center (y) of the fluence map is not at the isocenter", cntr_y)
        sys.exit()
    
    # To make the minimum value to be 0
    FluenceData = FluenceData - min(FluenceData)
    # Convert it into 2D array
    fluence_map = FluenceData.reshape(NrBixels.x, NrBixels.y)
    # This part needed to be checked!!!
    fluence_map = np.flipud(fluence_map)
    
    return fluence_map

def export_fluence(beam_set, export_path):
    if not os.path.exists(export_path):
        os.mkdir(export_path)
        
    for beam in beam_set.Beams:
        BixelWidth = beam.Fluence.BixelWidth
        Corner = beam.Fluence.Corner
        NrBixels = beam.Fluence.NrBixels
        FluenceData = beam.Fluence.FluenceData
        BeamIsoctr_temp = beam.Isocenter.Position
        
        # Convert BeamIsoctr to actual Dictionary datatype to save it as the nparray format
        BeamIsoctr = {}
        for i in BeamIsoctr_temp:
            BeamIsoctr[i] = BeamIsoctr_temp[i]

        # Getting Jaw positions for each beam
        jaws = export_jaws(beam)

        # Getting fluence map for each beam
        fluence_map = get_fluence_map(FluenceData, NrBixels, Corner, BixelWidth)
        fluence = np.array([beam.GantryAngle, BeamIsoctr, fluence_map, jaws], dtype=object)

        Name = str(format(beam.GantryAngle, ".1f")).replace('.', '_')
        fname = "Fluence_" + Name + ".npy"
        save_path = os.path.join(export_path, fname)
        np.save(save_path, fluence)

# Export Jaw positions
def export_jaws(Beam):
    Jaws = {}
    for idx, segment in enumerate(Beam.Segments):
        if idx == 0:
            Jaws['X1'] = round(segment.JawPositions[0], 2)
            Jaws['X2'] = round(segment.JawPositions[1], 2)
            Jaws['Y1'] = round(segment.JawPositions[2], 2)
            Jaws['Y2'] = round(segment.JawPositions[3], 2)
        else:
            X1 = round(segment.JawPositions[0], 2)
            X2 = round(segment.JawPositions[1], 2)
            Y1 = round(segment.JawPositions[2], 2)
            Y2 = round(segment.JawPositions[3], 2)

            if abs(X1 - Jaws['X1']) > 0.01:
                print("X1 is different from each segment ", X1, Jaws['X1'])
                sys.exit()
            if abs(X2 - Jaws['X2']) > 0.01:
                print("X2 is different from each segment ", X2, Jaws['X2'])
                sys.exit()
            if abs(Y1 - Jaws['Y1']) > 0.01:
                print("Y1 is different from each segment ", Y1, Jaws['Y1'])
                sys.exit()
            if abs(Y2 - Jaws['Y2']) > 0.01:
                print("Y2 is different from each segment ", Y2, Jaws['Y2'])
                sys.exit()

    return Jaws

def export_dcmimage(case, exam, export_path, plan_name):
    # Include External and other contours with override
    for roi in case.PatientModel.RegionsOfInterest:
        if roi.Type.lower() in ['external']:
            case.PatientModel.ToggleExcludeFromExport(ExcludeFromExport=False, RegionOfInterests=[roi.Name], PointsOfInterests=[])
        if hasattr(roi, 'OfRoi'):
            if hasattr(roi.OfRoi, 'RoiMaterial'):
                case.PatientModel.ToggleExcludeFromExport(ExcludeFromExport=False, RegionOfInterests=[roi.Name], PointsOfInterests=[])
    
    patient = get_current('Patient')
    patient.Save()
    
    path = os.path.join(export_path, plan_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print("DICOM image and RS exported to ", path)
        result = case.ScriptableDicomExport(ExportFolderPath=path,
                                            Examinations=[exam.Name],
                                            RtStructureSetsForExaminations=[exam.Name],
                                            DicomFilter="",
                                            IgnorePreConditionWarnings=True)
        
def export_RPRD(case, beamset, export_path):
    path = export_path + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Export DICOM files
    result = case.ScriptableDicomExport(ExportFolderPath=path,
                                        BeamSets=[beamset.BeamSetIdentifier()],
                                        #PhysicalBeamSetDoseForBeamSets=[beamset.BeamSetIdentifier()],
                                        PhysicalBeamDosesForBeamSets=[beamset.BeamSetIdentifier()],
                                        DicomFilter="",
                                        IgnorePreConditionWarnings=True)


    """
    result = case.ScriptableDicomExport(ExportFolderPath=export_path,
                                        Examinations=[examination.Name],
                                        RtStructureSetsForExaminations=[examination.Name],
                                        BeamSets=[beamset.BeamSetIdentifier()],
                                        PhysicalBeamSetDoseForBeamSets=[beamset.BeamSetIdentifier()],
                                        EffectiveBeamSetDoseForBeamSets=[beamset.BeamSetIdentifier()],
                                        PhysicalBeamDosesForBeamSets=[beamset.BeamSetIdentifier()],
                                        EffectiveBeamDosesForBeamSets=[beamset.BeamSetIdentifier()],
                                        DicomFilter="",
                                        IgnorePreConditionWarnings=True)
    """


def save_and_export(patient, case, beam_set, plan_path):  
    # Save the plan
    patient.Save()
    
    # Export RT plan and RT Dose files
    export_RPRD(case, beam_set, plan_path)
    
    # Export fluence
    #export_fluence(beam_set, plan_path)