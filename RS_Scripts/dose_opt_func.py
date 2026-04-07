# From RayStation 2025, connect module is replaced by rayastation module.
try:
    # Standard RayStation scripting environment
    from connect import *
    print("Using RayStation connect module")
except ImportError:
    # Alternative environment (e.g. research / RayStation Python API wrapper)
    from raystation import *
    print("Using raystation module")
import random as rand

# Checking if normal structure contours have non-zero volumes.
def check_volume_structures(plan):
    roi_to_check = ['Duodenum','Bowel_Small','Bowel_Large','Stomach','Liver','SpinalCord','Kidney_L','Kidney_R','Spleen','skin_auto']

    # Find all ROIs with defined geometry in the current beam set.
    #structure_set = get_current('BeamSet').GetStructureSet()

    structure_set = plan.BeamSets[0].GetStructureSet()
    roi_list = [r.OfRoi.Name for r in structure_set.RoiGeometries if r.PrimaryShape != None]
    
    nonzero_roi = []
    for roi in roi_to_check:
        if roi in roi_list:
            nonzero_roi.append(roi)
    
    return nonzero_roi

# Initialize the optimization variables
def initialize_opt(patient, plan, GTVName, TVIName, MaxNumItr=50, Tolerance=1e-5, DE_status=False):
    nonzero_roi = check_volume_structures(plan)

    # Delete all the existing goals and dose constraints
    delete_clinical_goals(plan)
    delete_dose_constraints(plan)
    
    # Add clinical goals
    add_clinical_goals(plan, nonzero_roi) # This is same for both DE and non-DE plans
    
    # Set optimization parameters
    set_optimization_param(plan, MaxNumItr, Tolerance)
    
    # Add dose constraints
    if DE_status == True:
        add_dose_constraints_DE(plan, GTVName, TVIName, nonzero_roi)
    else:
        add_dose_constraints_nonDE(plan, nonzero_roi)

    # Define the machine to be the latest 2100 machine (not supported by Raystation Scripting)
    """
    machine_db = get_current('MachineDB')
    machine_2107 = machine_db.QueryCommissionedMachineInfo(Filter={'Name':'Varian 2110_1'})
    retval_0 = case.AddNewPlan(PlanName="ttt", PlannedBy="", Comment="", ExaminationName="Primary Plan CT", IsMedicalOncologyPlan=False, AllowDuplicateNames=False)
    retval_0.AddNewBeamSet(Name="ttt", ExaminationName="Primary Plan CT", MachineName="Varian 2110_1", Modality="Photons", TreatmentTechnique="SMLC", PatientPosition="HeadFirstSupine", NumberOfFractions=5, CreateSetupBeams=False, UseLocalizationPointAsSetupIsocenter=False, UseUserSelectedIsocenterSetupIsocenter=False, Comment="", RbeModelName=None, EnableDynamicTrackingForVero=False, NewDoseSpecificationPointNames=[], NewDoseSpecificationPoints=[], MotionSynchronizationTechniqueSettings={ 'DisplayName': None, 'MotionSynchronizationSettings': None, 'RespiratoryIntervalTime': None, 'RespiratoryPhaseGatingDutyCycleTimePercentage': None }, Custom=None)
    """
    
    patient.Save()

# Set optimization segmentation parameters
def set_optimization_param(plan, MaxNumItr=80, Tolerance=1e-6):
    plan.PlanOptimizations[0].OptimizationParameters.Algorithm.MaxNumberOfIterations = MaxNumItr
    plan.PlanOptimizations[0].OptimizationParameters.Algorithm.OptimalityTolerance = Tolerance
    plan.PlanOptimizations[0].OptimizationParameters.DoseCalculation.ComputeFinalDose = True
    plan.PlanOptimizations[0].OptimizationParameters.DoseCalculation.ComputeIntermediateDose = False

# Update dose constraints
def update_dose_constraints(plan):   
    for cf in plan.PlanOptimizations[0].Objective.ConstituentFunctions:
        if cf.ForRegionOfInterest.OrganData.OrganType == 'Target':
            # random number between 0.8 to 1.2
            updated_param = round(1.0 + (0.4 * rand.random() - 0.2), 2)
            cf.DoseFunctionParameters.Weight = cf.DoseFunctionParameters.Weight * updated_param
        else:
            updated_param1 = round(1.0 + (0.4 * rand.random() - 0.2), 2)
            updated_param2 = round(1.0 + (0.4 * rand.random() - 0.2), 2)
            if hasattr(cf.DoseFunctionParameters, 'Weight'):
                cf.DoseFunctionParameters.Weight = cf.DoseFunctionParameters.Weight * updated_param1
            if hasattr(cf.DoseFunctionParameters, 'DoseLevel'):
                cf.DoseFunctionParameters.DoseLevel = cf.DoseFunctionParameters.DoseLevel * updated_param2


# escalate the dose constraint for the target if allowed
def run_optimization(plan, target_list=None, DE_PTV=None, dose_escal_step=500):
    # Update GTV constraints
    ConstituentFunctions = plan.PlanOptimizations[0].Objective.ConstituentFunctions
    if target_list is not None:
        for target in target_list:
            for obj in ConstituentFunctions:
                # If target name exists in the structure list and function type matched
                if obj.ForRegionOfInterest.Name == target and obj.DoseFunctionParameters.FunctionType == 'MinDvh': 
                    if hasattr(obj.DoseFunctionParameters, 'DoseLevel'):
                        Current_dose_level = obj.DoseFunctionParameters.DoseLevel
                        obj.DoseFunctionParameters.DoseLevel = Current_dose_level + dose_escal_step
                        print(obj.ForRegionOfInterest.Name, "dose level changed from ", Current_dose_level, obj.DoseFunctionParameters.DoseLevel)
    else: # When target is None, just optimze the plan without DE
        print("Simple plan optimization without DE")
    
    # Update DE-PTV constraints
    if DE_PTV is not None:
        for obj in ConstituentFunctions:
            # If target name exists in the structure list and maxdose constraint
            if obj.ForRegionOfInterest.Name == DE_PTV and obj.DoseFunctionParameters.FunctionType == 'MaxDose': 
                if hasattr(obj.DoseFunctionParameters, 'DoseLevel'):
                    Current_dose_level = obj.DoseFunctionParameters.DoseLevel
                    obj.DoseFunctionParameters.DoseLevel = Current_dose_level + (dose_escal_step * 1.10) # 110% max dose
                    print(obj.ForRegionOfInterest.Name, "dose level changed from ", Current_dose_level, obj.DoseFunctionParameters.DoseLevel)
    else: # When DE-PTV is None, just optimze the plan without changing in dose constraints
        print("Simple plan optimization without modifying dose constraints")
    
    # Optimize the plan
    plan.PlanOptimizations[0].RunOptimization()
    

# escalate the dose constraint for the target 
def run_dose_escalation(plan, target_list=['IGTV_pancreas', 'TVI'], DE_PTV='DE-PTV_auto', dose_escal_step=500):
    run_optimization(plan, target_list, DE_PTV, dose_escal_step)


# Check if clinical goal has met
def eval_clinical_goals(plan):  
    total_eval = True
    for eval in plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions:
        if eval.EvaluateClinicalGoal() == False:
            total_eval = False
            print("At least one of the constraints failed")

    return total_eval

# Check if clinical goal has met
def failed_clinical_goals(plan):  
    str_list = []
    for eval in plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions:
        if eval.EvaluateClinicalGoal() == False:
            str_list.append(eval.ForRegionOfInterest.Name)
            print(eval.ForRegionOfInterest.Name + " Failed to meet its clinical goals")

    return str_list

# Delete all the existing clinical goals
def delete_clinical_goals(plan):
    status = True
    while status == True:
        try:
            plan.TreatmentCourse.EvaluationSetup.DeleteClinicalGoal(
                FunctionToRemove=plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions[0])
        except:
            print("Deleted all the clinical goals")
            status = False


# Add the pre-defined clinical goals
def add_clinical_goals(plan, nonzero_roi):
    # Duodenum
    if 'Duodenum' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Duodenum",
                                                         GoalCriteria="AtMost",
                                                         GoalType="DoseAtAbsoluteVolume",
                                                         AcceptanceLevel=2000,
                                                         ParameterValue=30,
                                                         IsComparativeGoal=False,
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Duodenum",
                                                         GoalCriteria="AtMost",
                                                         GoalType="DoseAtAbsoluteVolume",
                                                         AcceptanceLevel=3500,
                                                         ParameterValue=1,
                                                         IsComparativeGoal=False,
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Duodenum", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3000, 
                                                         ParameterValue=10, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Duodenum", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=4000, 
                                                         ParameterValue=0, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
    # Small Bowel
    if "Bowel_Small" in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Small", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=2000, 
                                                         ParameterValue=30, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Small", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3500, 
                                                         ParameterValue=1, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Small", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3000, 
                                                         ParameterValue=10, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Small", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=4000, 
                                                         ParameterValue=0, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
        
    # Large Bowel
    if "Bowel_Large" in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Large", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=2000, 
                                                         ParameterValue=30, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Large", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3500, 
                                                         ParameterValue=1, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Large", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3000, 
                                                         ParameterValue=10, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Bowel_Large", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=4000, 
                                                         ParameterValue=0, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
    # Stomach
    if 'Stomach' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Stomach", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=2000, 
                                                         ParameterValue=30, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Stomach", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3500, 
                                                         ParameterValue=1, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Stomach", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=3000, 
                                                         ParameterValue=10, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Stomach", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=4000, 
                                                         ParameterValue=0, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)

    # Spinal Cord    
    if 'SpinalCord' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="SpinalCord", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtAbsoluteVolume", 
                                                         AcceptanceLevel=2000, 
                                                         ParameterValue=1, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)

    # Liver
    if 'Liver' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Liver", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtVolume", 
                                                         AcceptanceLevel=1200, 
                                                         ParameterValue=0.5, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
        
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Liver", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtVolume", 
                                                         AcceptanceLevel=5500, 
                                                         ParameterValue=0, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
    # Kidneys
    if 'Kidney_L' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Kidney_L", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtVolume", 
                                                         AcceptanceLevel=1200, 
                                                         ParameterValue=0.25, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)

    if 'Kidney_R' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="Kidney_R", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtVolume", 
                                                         AcceptanceLevel=1200, 
                                                         ParameterValue=0.25, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
    # skin_auto
    if 'skin_auto' in nonzero_roi:
        plan.TreatmentCourse.EvaluationSetup.AddClinicalGoal(RoiName="skin_auto", 
                                                         GoalCriteria="AtMost", 
                                                         GoalType="DoseAtVolume",
                                                         AcceptanceLevel=3500, 
                                                         ParameterValue=0, 
                                                         IsComparativeGoal=False, 
                                                         Priority=2147483647)
    
# Delete all the existing dose constrains
def delete_dose_constraints(plan):
    status = True
    while status == True:
        try:
            plan.PlanOptimizations[0].Objective.ConstituentFunctions[0].DeleteFunction()
        except:
            print("Deleted all the dose constraints")
            status = False

# Add dose constraints for dose escalation planning
def add_dose_constraints_DE(plan, GTVName, TVIName, nonzero_roi, initial_coverage=3800):         
    structure_set = plan.BeamSets[0].GetStructureSet()
    roi_list = [r.OfRoi.Name for r in structure_set.RoiGeometries if r.PrimaryShape != None]
    roi_volume = [r.GetRoiVolume() for r in structure_set.RoiGeometries if r.PrimaryShape != None and r.HasContours() == True]
    
    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MinDvh", RoiName=GTVName, IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[0].DoseFunctionParameters.DoseLevel = initial_coverage
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[0].DoseFunctionParameters.PercentVolume = 98
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[0].DoseFunctionParameters.Weight = 50

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MinDvh", RoiName="DE-PTV_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[1].DoseFunctionParameters.DoseLevel = initial_coverage
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[1].DoseFunctionParameters.PercentVolume = 98
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[1].DoseFunctionParameters.Weight = 100

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="DE-PTV_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[2].DoseFunctionParameters.DoseLevel = round(initial_coverage * 1.25)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[2].DoseFunctionParameters.Weight = 10

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MinDvh", RoiName=TVIName, IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[3].DoseFunctionParameters.DoseLevel = initial_coverage
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[3].DoseFunctionParameters.PercentVolume = 98
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[3].DoseFunctionParameters.Weight = 50

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="GI-PRV_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[4].DoseFunctionParameters.DoseLevel = 3300
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[4].DoseFunctionParameters.Weight = 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="GI-PRV_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[5].DoseFunctionParameters.DoseLevel = 3000
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[5].DoseFunctionParameters.PercentVolume = 5

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="fsNT10mm_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[6].DoseFunctionParameters.DoseLevel = 1800
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[6].DoseFunctionParameters.Weight = 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="fsNT_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[7].DoseFunctionParameters.DoseLevel = 2500
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[7].DoseFunctionParameters.Weight = 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="DoseFallOff", RoiName="fsbody_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[8].DoseFunctionParameters.HighDoseLevel = 3000
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[8].DoseFunctionParameters.LowDoseLevel = 1500
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[8].DoseFunctionParameters.LowDoseDistance = 8
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[8].DoseFunctionParameters.Weight = 20

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="fsTargetRing3mm_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[9].DoseFunctionParameters.DoseLevel = 3500
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[9].DoseFunctionParameters.PercentVolume = 3

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="fsTargetRing7mm_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[10].DoseFunctionParameters.DoseLevel = 3000
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[10].DoseFunctionParameters.PercentVolume = 5

    idx_param = 11
    if 'Stomach' in nonzero_roi:
        vol_stomach = roi_volume[roi_list.index('Stomach')]
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="Stomach", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 50
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Stomach", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3400
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(2./vol_stomach),3) # 2cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 10
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Stomach", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 2900
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(10./vol_stomach),3) # 10cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Stomach", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1900
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(30./vol_stomach),3) # 30cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1

    if 'Bowel_Small' in nonzero_roi:
        vol_sb = roi_volume[roi_list.index('Bowel_Small')]
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="Bowel_Small", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 50
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Bowel_Small", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3400
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(2./vol_sb),3) # 2cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 10
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Bowel_Small", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 2900
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(10./vol_sb),3) # 10cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Bowel_Small", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(30./vol_sb),3) # 30cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1

    if 'Duodenum' in nonzero_roi:
        vol_duodenum = roi_volume[roi_list.index('Duodenum')]
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="Duodenum", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 50
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Duodenum", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3400
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(2./vol_duodenum),3) # 2cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 10
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Duodenum", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 2900
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(10./vol_duodenum),3) # 10cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Duodenum", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(30./vol_duodenum),3) # 30cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1
  
    if 'Bowel_Large' in nonzero_roi:
        vol_lb = roi_volume[roi_list.index('Bowel_Large')]
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="Bowel_Large", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 50
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Bowel_Large", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3400
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(2./vol_lb),3) # 2cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 10
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Bowel_Large", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 2900
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(10./vol_lb),3) # 10cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Bowel_Large", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = round(100*(30./vol_lb),3) # 30cc in percentage
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 5
        idx_param += 1
    
    if 'fsOAR-PTVoverlap_auto' in roi_list:
        if structure_set.RoiGeometries['fsOAR-PTVoverlap_auto'].HasContours() and structure_set.RoiGeometries['fsOAR-PTVoverlap_auto'].GetRoiVolume() > 0.005:
            plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="fsOAR-PTVoverlap_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
            plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 3700
            plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 25
            idx_param += 1

    if 'Kidney_R' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Kidney_R", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1200
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = 20
        idx_param += 1

    if 'Kidney_L' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Kidney_L", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1200
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = 20
        idx_param += 1

    if 'Liver' in nonzero_roi:
        vol_sb = roi_volume[roi_list.index('Liver')]
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Liver", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 1200
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.PercentVolume = 40
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 1
        idx_param += 1
        
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="Liver", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 5300
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 10
        idx_param += 1
    
    if 'Spleen' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxEud", RoiName="Spleen", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 800
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 1
        idx_param += 1
        
    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="skin_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.DoseLevel = 2800
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx_param].DoseFunctionParameters.Weight = 30
    idx_param += 1


# Add dose constraints for non-dose escalation planning
# Not used anymore since nonDE plans are not re-optimized
def add_dose_constraints_nonDE(plan, nonzero_roi):
    idx = 0
    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="GI-PRV_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 3300
    idx += 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="GI-PRV_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 3000
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 5
    idx += 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="fsNT10mm_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 1800
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 40
    idx += 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="fsNT_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 2500
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 40
    idx += 1

    """
    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MinDvh", RoiName="PTV25_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 2500
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 95
    idx += 1

    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MinDvh", RoiName="PTV40_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 4000
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 95
    idx += 1
    """
    
    plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="skin_auto", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 2800
    plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 0
    idx += 1
    
    if 'Bowel_Large' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName="Bowel_Large", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 3000
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 5
        idx += 1
        
    if 'Kidney_R' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Kidney_R", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 1200
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 20
        idx += 1

    if 'Kidney_L' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Kidney_L", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 1200
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 20
        idx += 1

    if 'Liver' in nonzero_roi:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDvh", RoiName="Liver", IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.DoseLevel = 1200
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[idx].DoseFunctionParameters.PercentVolume = 40
        idx += 1
        
# Add new beams
def add_9beams(beam_set, poi_pos):
    iso_x = poi_pos.x
    iso_y = poi_pos.y
    iso_z = poi_pos.z

    retval = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="A_000", Description="", GantryAngle=0, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retval.SetBolus(BolusName="")
    beam_set.Beams['A_000'].BeamMU = 0

    retva2 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="B_040", Description="", GantryAngle=40, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva2.SetBolus(BolusName="")
    beam_set.Beams['B_040'].BeamMU = 0

    retva3 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="C_080", Description="", GantryAngle=80, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva3.SetBolus(BolusName="")
    beam_set.Beams['C_080'].BeamMU = 0

    retva4 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="D_120", Description="", GantryAngle=120, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva4.SetBolus(BolusName="")
    beam_set.Beams['D_120'].BeamMU = 0

    retva5 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="E_160", Description="", GantryAngle=160, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva5.SetBolus(BolusName="")
    beam_set.Beams['E_160'].BeamMU = 0

    retva6 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="F_200", Description="", GantryAngle=200, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva6.SetBolus(BolusName="")
    beam_set.Beams['F_200'].BeamMU = 0

    retva7 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="G_240", Description="", GantryAngle=240, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva7.SetBolus(BolusName="")
    beam_set.Beams['G_240'].BeamMU = 0

    retva8 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="H_280", Description="", GantryAngle=280, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva8.SetBolus(BolusName="")
    beam_set.Beams['H_280'].BeamMU = 0

    retva9 = beam_set.CreatePhotonBeam(BeamQualityId="6", CyberKnifeCollimationType="Undefined", CyberKnifeNodeSetName=None, CyberKnifeNodeSetCommissionTime=None, IsocenterData={ 'Position': { 'x': iso_x, 'y': iso_y, 'z': iso_z }, 'NameOfIsocenterToRef': "Tx Iso", 'Name': "Tx Iso", 'Color': "Yellow" }, Name="I_320", Description="", GantryAngle=320, CouchRotationAngle=0, CouchPitchAngle=0, CouchRollAngle=0, CollimatorAngle=0)
    retva9.SetBolus(BolusName="")
    beam_set.Beams['I_320'].BeamMU = 0
    
def delete_beams(beam_set):
    for beam in beam_set.Beams:
        beam_set.DeleteBeam(BeamName=beam.Name)
        
def hotspot_reduction(case, plan, examination, idx, threshold=3800):
    failed_list = failed_clinical_goals(plan)
    structure_set = plan.BeamSets[0].GetStructureSet()
    str_list = []
    
    Rois = case.PatientModel.StructureSets[examination.Name].RoiGeometries
    for structure in Rois:
        str_list.append(structure.OfRoi.Name)
    
    margin1 = 0.1
    margin2 = 0.1
    
    fsIDL_name = 'fs' + str(threshold) + 'cGy_auto' + str(idx)
    Hotspots_name = 'Hotspots_auto' + str(idx)
    # Create a new ROI and create its geometry from the plan dose and the threshold level
    plan_dose = plan.TreatmentCourse.TotalDose
    if fsIDL_name not in str_list:
        case.PatientModel.CreateRoi(Name = fsIDL_name, Color = 'Orange', Type = 'Control')
    # Threshold 
    case.PatientModel.RegionsOfInterest[fsIDL_name].CreateRoiGeometryFromDose(DoseDistribution = plan_dose, ThresholdLevel = threshold)
    
    # create the structure that overlaps the 35/38Gy isodose volume with the GI-PRV_auto
    if Hotspots_name not in str_list:
    	case.PatientModel.CreateRoi(Name=Hotspots_name, Color="Red", Type="Control", TissueName=None, RbeCellTypeName=None, RoiMaterial=None)
    case.PatientModel.RegionsOfInterest[Hotspots_name].SetAlgebraExpression(ExpressionA={ 'Operation': "Union", 'SourceRoiNames': [fsIDL_name], 'MarginSettings': { 'Type': "Expand", 'Superior': margin1, 'Inferior': margin1, 'Anterior': margin1, 'Posterior': margin1, 'Right': margin1, 'Left': margin1 } }, ExpressionB={ 'Operation': "Union", 'SourceRoiNames': ["GI-PRV_auto"], 'MarginSettings': { 'Type': "Expand", 'Superior': margin2, 'Inferior': margin2, 'Anterior': margin2, 'Posterior': margin2, 'Right': margin2, 'Left': margin2 } }, ResultOperation="Intersection", ResultMarginSettings={ 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 })
    case.PatientModel.RegionsOfInterest[Hotspots_name].UpdateDerivedGeometry(Examination=examination, Algorithm="Auto")
    
    # Increase the margin for hotspot structure
    if structure_set.RoiGeometries[Hotspots_name].HasContours() == True:
        while structure_set.RoiGeometries[Hotspots_name].GetRoiVolume() < 0.005:
            margin1 += 0.05
            margin2 += 0.05
            case.PatientModel.RegionsOfInterest[Hotspots_name].SetAlgebraExpression(ExpressionA={ 'Operation': "Union", 'SourceRoiNames': [fsIDL_name], 'MarginSettings': { 'Type': "Expand", 'Superior': margin1, 'Inferior': margin1, 'Anterior': margin1, 'Posterior': margin1, 'Right': margin1, 'Left': margin1 } }, ExpressionB={ 'Operation': "Union", 'SourceRoiNames': ["GI-PRV_auto"], 'MarginSettings': { 'Type': "Expand", 'Superior': margin2, 'Inferior': margin2, 'Anterior': margin2, 'Posterior': margin2, 'Right': margin2, 'Left': margin2 } }, ResultOperation="Intersection", ResultMarginSettings={ 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 })
            case.PatientModel.RegionsOfInterest[Hotspots_name].UpdateDerivedGeometry(Examination=examination, Algorithm="Auto")
            
        print("Margin for the Bowel Hotspots contour : ", margin1)
        
        # Add constraints for the new hotspot structure
        index = 0
        obj_list = []
        for cf in plan.PlanOptimizations[0].Objective.ConstituentFunctions:
            index += 1
            obj_list.append(cf.ForRegionOfInterest.Name)
        
        if Hotspots_name not in obj_list:
            plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName=Hotspots_name, IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
            plan.PlanOptimizations[0].Objective.ConstituentFunctions[index].DoseFunctionParameters.DoseLevel = 3500
            plan.PlanOptimizations[0].Objective.ConstituentFunctions[index].DoseFunctionParameters.Weight = 100

    # Add hotspot constraints in the liver as well
    if 'Liver' in failed_list:
        add_reduction_str(str_list, case, plan, examination, 'Liver', 1, 5300, 200)
    if 'skin_auto' in failed_list:
        add_reduction_str(str_list, case, plan, examination, 'skin_auto', 1, 3200, 150)

    run_optimization(plan, target_list=None, DE_PTV=None, dose_escal_step=0)
    

def add_reduction_str(str_list, case, plan, examination, str_name, idx, threshold=5300, reduction=300):
    structure_set = plan.BeamSets[0].GetStructureSet()
    margin1 = 0.1
    margin2 = 0.1
    
    fsIDL_name = 'fs' + str_name + str(threshold) + 'cGy_auto' + str(idx)
    Hotspots_name = 'Hotspots_' + str_name + '_auto' + str(idx)
    # Create a new ROI and create its geometry from the plan dose and the threshold level
    plan_dose = plan.TreatmentCourse.TotalDose
    if fsIDL_name not in str_list:
        case.PatientModel.CreateRoi(Name = fsIDL_name, Color = 'Orange', Type = 'Control')
    # Threshold 
    case.PatientModel.RegionsOfInterest[fsIDL_name].CreateRoiGeometryFromDose(DoseDistribution = plan_dose, ThresholdLevel = threshold)
    
    # create the structure that overlaps the threshold isodose volume with the ROI
    if Hotspots_name not in str_list:
    	case.PatientModel.CreateRoi(Name=Hotspots_name, Color="Red", Type="Control", TissueName=None, RbeCellTypeName=None, RoiMaterial=None)

    case.PatientModel.RegionsOfInterest[Hotspots_name].SetAlgebraExpression(ExpressionA={ 'Operation': "Union", 'SourceRoiNames': [fsIDL_name], 'MarginSettings': { 'Type': "Expand", 'Superior': margin1, 'Inferior': margin1, 'Anterior': margin1, 'Posterior': margin1, 'Right': margin1, 'Left': margin1 } }, ExpressionB={ 'Operation': "Union", 'SourceRoiNames': [str_name], 'MarginSettings': { 'Type': "Expand", 'Superior': margin2, 'Inferior': margin2, 'Anterior': margin2, 'Posterior': margin2, 'Right': margin2, 'Left': margin2 } }, ResultOperation="Intersection", ResultMarginSettings={ 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 })
    case.PatientModel.RegionsOfInterest[Hotspots_name].UpdateDerivedGeometry(Examination=examination, Algorithm="Auto")
    
    # Increase the margin for hotspot structure if too small
    while structure_set.RoiGeometries[Hotspots_name].GetRoiVolume() < 0.005:
        margin1 += 0.05
        margin2 += 0.05
        case.PatientModel.RegionsOfInterest[Hotspots_name].SetAlgebraExpression(ExpressionA={ 'Operation': "Union", 'SourceRoiNames': [fsIDL_name], 'MarginSettings': { 'Type': "Expand", 'Superior': margin1, 'Inferior': margin1, 'Anterior': margin1, 'Posterior': margin1, 'Right': margin1, 'Left': margin1 } }, ExpressionB={ 'Operation': "Union", 'SourceRoiNames': [str_name], 'MarginSettings': { 'Type': "Expand", 'Superior': margin2, 'Inferior': margin2, 'Anterior': margin2, 'Posterior': margin2, 'Right': margin2, 'Left': margin2 } }, ResultOperation="Intersection", ResultMarginSettings={ 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 })
        case.PatientModel.RegionsOfInterest[Hotspots_name].UpdateDerivedGeometry(Examination=examination, Algorithm="Auto")
        
    print("Margin for the" + str_name + "Hotspots contour : ", margin1)
    
    # Add constraints for the new hotspot structure
    index = 0
    obj_list = []
    for cf in plan.PlanOptimizations[0].Objective.ConstituentFunctions:
        index += 1
        obj_list.append(cf.ForRegionOfInterest.Name)
    
    if Hotspots_name not in obj_list:
        plan.PlanOptimizations[0].AddOptimizationFunction(FunctionType="MaxDose", RoiName=Hotspots_name, IsConstraint=False, RestrictAllBeamsIndividually=False, RestrictToBeam=None, IsRobust=False, RestrictToBeamSet=None, UseRbeDose=False)
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[index].DoseFunctionParameters.DoseLevel = threshold - reduction
        plan.PlanOptimizations[0].Objective.ConstituentFunctions[index].DoseFunctionParameters.Weight = 100