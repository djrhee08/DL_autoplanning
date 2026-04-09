# From RayStation 2025, connect module is replaced by rayastation module.
try:
    # Standard RayStation scripting environment
    from connect import *
    print("Using RayStation connect module")
except ImportError:
    # Alternative environment (e.g. research / RayStation Python API wrapper)
    from raystation import *
    print("Using raystation module")

import random
import string
import os, sys
from export_dcm import export_dcmimage, save_and_export
from dose_opt_func import set_optimization_param, update_dose_constraints
from fetch_info import GetPatientInfo, Load_Patient
from reset_iso import get_newiso

def update_collimator_angle(beam_set, idx, new_collimator_angle):
    beam = beam_set.Beams[idx]
    IsocenterData = {'Position':beam.Isocenter.Position, 
                        'Name':beam.Isocenter.Annotation.Name,
                        'NameOfIsocenterToRef':beam.Isocenter.Annotation.Name,
                        'Color':beam.Isocenter.Annotation.DisplayColor
                    }
    beam_params = {
        'Name': beam.Name,
        'Description': beam.Description,
        'GantryAngle': beam.GantryAngle,
        'CollimatorAngle': new_collimator_angle,
        'IsocenterData': IsocenterData,
        'BeamQualityId': beam.BeamQualityId,
        'CouchRotationAngle': beam.CouchRotationAngle,
        'CouchPitchAngle': beam.CouchPitchAngle,
        'CouchRollAngle': beam.CouchRollAngle,
    }
    beam_set.DeleteBeam(BeamName=beam.Name)
    beam_set.CreatePhotonBeam(**beam_params)


def generate_vmat_range():
    """
    Generates a random VMAT gantry angle range (start, end)
    """
    # Assuming the gantry rotation is always in clockwise
    # create a gantry range between 180 and 360
    gant_range = random.randint(180, 360)
    gant_start = random.randint(-179, 0)
    # Add random number between 90 to 180 to the gantry start, assuming that gantry rotation direction doesn't matter
    # minimum gantry range becomes 90
    gant_end = gant_start + gant_range
    if gant_end > 179:
        gant_end = 179

    # Make the gantry range to be an even number
    if (gant_end - gant_start) % 2 != 0:
        gant_end -= 1

    gant_start += 360 

    return gant_start, gant_end


# Exclude auto structures from export & update derived geometries
def update_rois(case, exam, is_exclusion=False):
    ROI_exclusion_list = []
    ROIs = case.PatientModel.RegionsOfInterest
    for roi in ROIs:
        if 'auto' in roi.Name.lower():
            ROI_exclusion_list.append(roi.Name)
        if roi.DerivedRoiExpression is not None:
            case.PatientModel.RegionsOfInterest[roi.Name].UpdateDerivedGeometry(Examination=exam, Algorithm="Auto")

    if is_exclusion:
        case.PatientModel.ToggleExcludeFromExport(ExcludeFromExport=True, RegionOfInterests=ROI_exclusion_list, PointsOfInterests=[])


def update_objs(case, plan, CTName):
    print("Updating objectives....")
    for constfunc in plan.PlanOptimizations[0].Objective.ConstituentFunctions:
        roi = constfunc.ForRegionOfInterest.Name
        roigeom = case.PatientModel.StructureSets[CTName].RoiGeometries[roi]
        print("roi : ", roi)
        if roigeom.HasContours() == False:
            constfunc.DeleteFunction()
            print("roi obj deleted!")

def delete_all_beams(beam_set):
    # Delete all beams in the beam set
    # Need to run two for loops, otherwise, the order is messed up
    beam_list = []
    for beam in beam_set.Beams:
        beam_list.append(beam.Name)
    for BeamName in beam_list:
        beam_set.DeleteBeam(BeamName=BeamName)
        print("Beam deleted: ", BeamName)

def delete_normalization(case):
    # Delete normalization for all plans
    for plan in case.TreatmentPlans:
        for beamset in plan.BeamSets:
            if hasattr(beamset.Prescription, 'PrescriptionDoseReferences') == True:
                if len(beamset.Prescription.PrescriptionDoseReferences) > 0:
                    beamset.DeletePrimaryPrescription()

def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



num_arcs = 1 # Number of VMAT arcs
Site = 'total' # Thor, CNS, pancreas, HN, GI, GU
cwd = os.getcwd()
export_path = os.path.join(cwd, "data", Site)
characters = string.ascii_letters + string.digits

CTTableName = "MDACC CT Table_1"
pID_list = []
LastName_list = []
PlanName_list = []
CaseName_list = []

# Total number of plan generated for data creation 
num_itr = 20 # Number of VMAT plans derived from the original plan with some variations
is_coll = False # True if collimator angle is random, False if collimator angle is fixed to be 0

# Segment area and number of segments for each plan
MinSegmentArea = 4 # default 4 cm^2
MaxNumberOfSegments = 100 # default 50
MinSegmentMUPerFraction = 2 # default 2

with open('patient_list_' + Site + '.dat') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        temp = line.rstrip('\n').split('\t')
        pID_list.append(temp[0])
        LastName_list.append(temp[1])
        CaseName_list.append(temp[2])
        PlanName_list.append(temp[3])


for idx, pID in enumerate(pID_list):
    LastName = LastName_list[idx]
    PlanName = PlanName_list[idx]
    CaseName = CaseName_list[idx]
    
    patient = Load_Patient(pID, LastName)
    case = patient.Cases[CaseName]
    print(LastName, pID)
    delete_normalization(case)
    
    plan = case.TreatmentPlans[PlanName]
    
    # Check if the original plan is not VMAT, need to create an arbitrary VMAT plan
    if plan.BeamSets[0].DeliveryTechnique != 'DynamicArc':
        print(plan.BeamSets[0].DeliveryTechnique)
        print("This is not a VMAT patient")
        sys.exit()
        
    # update the minsegarea and maxnumofsegments in the original plan
    plan.PlanOptimizations[0].OptimizationParameters.TreatmentSetupSettings[0].SegmentConversion.MinSegmentArea = MinSegmentArea
    plan.PlanOptimizations[0].OptimizationParameters.TreatmentSetupSettings[0].SegmentConversion.MaxNumberOfSegments = MaxNumberOfSegments
    plan.PlanOptimizations[0].OptimizationParameters.TreatmentSetupSettings[0].SegmentConversion.MinSegmentMUPerFraction = MinSegmentMUPerFraction
    
    # Remove setup beams
    plan.BeamSets[0].RemoveSetupBeams()
    patient.Save()

    # Get information from the patient + Assign CT Table
    CTName = GetPatientInfo(pID, LastName, case, plan, CTTableName)
    exam = case.Examinations[CTName]
    beam_set = plan.BeamSets[0]
    
    # Create a patient_plan directory
    CT_path = os.path.join(export_path, pID)
    if not os.path.exists(CT_path):
        os.makedirs(CT_path)

    update_rois(case, exam, is_exclusion=True)
    update_objs(case, plan, CTName)
    patient.Save()

    # Export CT image + RTS file
    export_dcmimage(case, exam, CT_path, plan_name='CT')
    max_retries = 3


    for itr in range(num_itr):
        # Copy the plan for data generation
        NewPlanName = "VMAT_" + str(num_arcs) + "arcs_" + str(itr).zfill(2)
        case.CopyPlan(PlanName=PlanName, NewPlanName=NewPlanName, KeepBeamSetNames=True)
        
        # Get Plan and BeamSets from the copied plan
        plan = case.TreatmentPlans[NewPlanName]
        beam_set = plan.BeamSets[0]
        current_iso = beam_set.Beams[0].Isocenter.Position
        
        # Create a patient directory to save RD, RP files
        plan_path = os.path.join(CT_path, NewPlanName)
        if not os.path.exists(plan_path):
            os.makedirs(plan_path)
        
        # Reset optimization
        plan.PlanOptimizations[0].ResetOptimization()
        
        # Try this 3 times if not working
        for attempt in range(max_retries):
            try:
                # Create random numbers for Gantry and Collimator angles
                gant_angle = []
                coll_angle = []
                
                for idx2 in range(num_arcs):
                    """ Gantry angle generation for non-full arcs. May use it later, but not for now
                    if itr == 0:
                        # For the first iteration, create a fixed full gantry angle
                        start = 181
                        end = 179
                    else:
                        start, end = generate_vmat_range()
                    """
                    # Generate full arcs only for now
                    # When odd # of arcs, one arc starts from 181 to 179, the other arc starts from 179 to 181
                    if idx2 % 2 == 0:
                        start = 181
                        end = 179
                    elif idx2 % 2 == 1:
                        start = 179
                        end = 181
                    
                    gant_angle.append([start, end])
                    
                    # create a random number in between 0 and 90 / 270 and 360
                    if is_coll == True:
                        rand_num2 = random.choice([random.randint(0, 90), random.randint(270, 360)])
                        while rand_num2 in coll_angle:
                            rand_num2 = random.choice([random.randint(0, 90), random.randint(270, 360)])
                        coll_angle.append(rand_num2)
                    else:
                        # collimator angle is fixed to be 0
                        coll_angle.append(0)

                print("new gantry angle range:", gant_angle)
                print("new collimator angles:", coll_angle)
                
                # Get new isocenter coordinates
                rand_IsoName = random_string = ''.join(random.choices(characters, k=5))
                iso_x, iso_y, iso_z = get_newiso(case, CTName, current_iso)
                print("new isocenter:", iso_x, iso_y, iso_z)
                
                print("Deleting existing beams... ")
                delete_all_beams(beam_set)

                print("Creating new arcs...")
                beam_set = plan.BeamSets[0]
                
                # Create arcs based on the given gantry ang collimator angles, and new isocenter coordinates
                for idx3 in range(num_arcs):
                    if idx3 % 2 == 0:
                        ArcRotationDirection = "Clockwise"
                    elif idx3 % 2 == 1:
                        ArcRotationDirection = "CounterClockwise"
                    
                    beam_set.CreateArcBeam(ArcStopGantryAngle=gant_angle[idx3][1], 
                                            ArcRotationDirection=ArcRotationDirection, 
                                            BeamQualityId="6", 
                                            IsocenterData={'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 
                                                            'NameOfIsocenterToRef': rand_IsoName, 
                                                            'Name': rand_IsoName, 
                                                            'Color': "98, 184, 234" }, 
                                            Name=string.ascii_letters[26+idx3], Description="", 
                                            GantryAngle=gant_angle[idx3][0], 
                                            CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, 
                                            CollimatorAngle=coll_angle[idx3])
                
                # Make the gantry spacing to be 2
                for iii in range(num_arcs):
                    plan.PlanOptimizations[0].OptimizationParameters.TreatmentSetupSettings[0].BeamSettings[iii].ArcConversionPropertiesPerBeam.EditArcBasedBeamOptimizationSettings(CreateDualArcs=False, FinalGantrySpacing=2, MaxArcDeliveryTime=90, BurstGantrySpacing=None, MaxArcMU=None)
                
                print("New arcs are created")
                
                update_rois(case, exam, is_exclusion=False)
                print("rois updated")

                # Reoptimize the field with random # of iteration
                rand_itr = random.randrange(40, 80)
                set_optimization_param(plan, MaxNumItr=rand_itr, Tolerance=1e-6)
                update_dose_constraints(plan)
                print("optimization parameter set")
                
                plan.PlanOptimizations[0].ResetOptimization()
                plan.PlanOptimizations[0].RunOptimization()
                print("optimization done")
                
                beam_set = plan.BeamSets[0]
                save_and_export(patient, case, beam_set, plan_path)
                print("save and exported")
                
                break
            except Exception as e:
                # Convert the exception to a string to check its contents
                error_message = str(e)
                
                # Log the specific MLC error if it occurs
                if "Leaves cannot close within the field" in error_message:
                    print(f"[WARNING] Skipping {NewPlanName}: The field is too wide for the MLCs to close.")
                else:
                    # Log any other unexpected errors
                    print(f"[ERROR] Skipping {NewPlanName} due to an unexpected optimization failure: {error_message}")
            finally:
                # Optional: Clean up logic that should happen whether it succeeded or failed
                # e.g., closing the patient to free up memory before the next loop
                # patient.Close()
                pass