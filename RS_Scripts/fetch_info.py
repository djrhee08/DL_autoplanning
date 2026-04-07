# From RayStation 2025, connect module is replaced by rayastation module.
try:
    # Standard RayStation scripting environment
    from connect import *
    print("Using RayStation connect module")
except ImportError:
    # Alternative environment (e.g. research / RayStation Python API wrapper)
    from raystation import *
    print("Using raystation module")
import sys

def Load_Patient(pID, LastName):
    # Convert pID into a string variable
    if not isinstance(pID, str):
        pID = str(pID)
        
    LastName = '^' + LastName + '$'
    patient_db = get_current("PatientDB")
    info = patient_db.QueryPatientInfo(Filter={'PatientID':pID, 
                                               'LastName':LastName})
    
    if len(info) == 1:
        patient = patient_db.LoadPatient(PatientInfo=info[0])
    elif len(info) == 0:
        raise Exception("There is no patient with the current pID")
    else:
        raise Exception("More than one patient with the same pID exists")
    
    # Load the first case in the patient
    return patient

def GetPatientInfo(pID, LastName, case, plan, CTTableName):
    Load_Patient(pID, LastName)

    # Get the list of exams with 'CTOR'
    CTName = plan.BeamSets[0].GetPlanningExamination().Name
    
    print("CTName : ", CTName)
            
    # assign a proper CT Table to the image
    examination = case.Examinations[CTName]
    
    #if examination.EquipmentInfo.ImagingSystemReference.ImagingSystemName != CTTableName:
    if hasattr(examination.EquipmentInfo.ImagingSystemReference, 'ImagingSystemName') == False:
        examination.EquipmentInfo.SetImagingSystemReference(ImagingSystemName=CTTableName)
    
    return CTName