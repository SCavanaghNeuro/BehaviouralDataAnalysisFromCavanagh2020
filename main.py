import scipy.io as spio
import numpy as np
import SeanFunc as SF
from os.path import dirname,realpath, join as pjoin

dir_path = dirname(realpath(__file__))
DataFileLocations = dir_path + '/DataFiles'
FilesNames = ['NonDrugDayProcessedData_SubjectH_63_Sessions21-Aug-2020.mat','NonDrugDayProcessedData_SubjectA_41_Sessions21-Aug-2020.mat']
Subjects =['Monkey H','Monkey A']

#pre-assign empty variables for useful outputs to be stored
EvidenceUnitsA_acrossSessions, EvidenceUnitsB_acrossSessions, Choices_acrossSessions,LongSampleTrial, \
    resp_trials,TrialType,DiscardTrials,TrialError,TrialErCompTr,IndexOfTrToUseFromRT,IndexOfTrToUseFromAllTr,CV_OutputValues,libDD =\
    [None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2
print('Analysing standard day sessions')

for i in [0,1]: #loop across the two subjects
    print('     Processing data for ' + Subjects[i])

    #Load the matlab data
    lib = spio.loadmat(pjoin(DataFileLocations,FilesNames[i]))

    #Extract the key variables from matlab structure
    # Reorganise the variables to collapse the data across sessions
    EvidenceUnitsA_acrossSessions[i] = np.concatenate((lib['DataStructureSessions']['EvidenceUnitsA'][0,:]),axis=0)
    EvidenceUnitsB_acrossSessions[i] = np.concatenate((lib['DataStructureSessions']['EvidenceUnitsB'][0,:]),axis=0)
    Choices_acrossSessions[i] = np.concatenate((lib['DataStructureSessions']['ChosenTarget'][0, :]), axis=1).reshape(-1)==1
    LongSampleTrial[i] = np.concatenate((lib['DataStructureSessions']['LongSampleTrial'][0, :]), axis=1).reshape(-1)
    resp_trials[i] = np.concatenate((lib['DataStructureSessions']['resp_trials'][0, :]), axis=0).reshape(-1)
    TrialType[i] = np.concatenate((lib['DataStructureSessions']['TrialType'][0, :]), axis=1).reshape(-1)
    DiscardTrials[i] = np.concatenate((lib['DataStructureSessions']['DiscardTrials'][0, :]), axis=1).reshape(-1)
    TrialError[i] = np.concatenate((lib['DataStructureSessions']['TrialError'][0, :]), axis=0).reshape(-1)

    #Select trials to use for analysisOnly use long trials, which are regular or NarrowBroad
    TrialErCompTr[i] = TrialError[i][resp_trials[i]==1]
    IndexOfTrToUseFromAllTr[i] = (LongSampleTrial[i]==1)*(resp_trials[i]==1)*(((TrialType[i]>17)*(TrialType[i]<24))+(TrialType[i]==1))
    IndexOfTrToUseFromRT[i] = IndexOfTrToUseFromAllTr[i][resp_trials[i]==1]

    #Figure 2 from paper: psychometric function and weighting across time
    MethodInput = 'CorrectIncorrect'
    SF.MakeFig2(IndexOfTrToUseFromRT[i],EvidenceUnitsA_acrossSessions[i],EvidenceUnitsB_acrossSessions[i],Choices_acrossSessions[i],TrialErCompTr[i],MethodInput,Subjects[i])

    #Supplementary tables from paper: Further regression analyses and model comparisson
    BootStrapNo=0 #fit the regression with a lapse term (only relevant for subsequent analyses so set to 0 here)
    n_kcv_runs = 2 # How many cross-validation runs to run? Note, in the paper, this was set to 100. It has been set to 2 here to decrease computing time.
    CV_OutputValues[i] = SF.AdvancedRegrAnalysis(Choices_acrossSessions[i][IndexOfTrToUseFromRT[i]][TrialType[i][IndexOfTrToUseFromAllTr[i]]==1],\
        EvidenceUnitsA_acrossSessions[i][IndexOfTrToUseFromRT[i],:][TrialType[i][IndexOfTrToUseFromAllTr[i]]==1,:],
        EvidenceUnitsB_acrossSessions[i][IndexOfTrToUseFromRT[i], :][TrialType[i][IndexOfTrToUseFromAllTr[i]] == 1, :],
                                           BootStrapNo,n_kcv_runs)

#Collapse some of the key variables across subjects so they can be used in further analyses
ChoicesCollapsed = np.concatenate(Choices_acrossSessions,axis=0)[np.concatenate(IndexOfTrToUseFromRT,axis=0)]
TrTypeCollapsed = np.concatenate(TrialType,axis=0)[np.concatenate(IndexOfTrToUseFromAllTr,axis=0)]
EvidUnitsACollapsed = np.concatenate(EvidenceUnitsA_acrossSessions,axis=0)[np.concatenate(IndexOfTrToUseFromRT,axis=0)]
EvidUnitsBCollapsed = np.concatenate(EvidenceUnitsB_acrossSessions,axis=0)[np.concatenate(IndexOfTrToUseFromRT,axis=0)]
TrErCollapsed = np.concatenate(TrialErCompTr[:],axis=0)[np.concatenate(IndexOfTrToUseFromRT,axis=0)]

#Figure 3 from paper: Subjects show a pro-variance bias in their choices on Narrow-Broad Trials
SF.Analyse_NarBroad(ChoicesCollapsed,TrTypeCollapsed,EvidUnitsACollapsed,EvidUnitsBCollapsed)

#Figure 4 from paper: Subjects show a pro-variance bias in their choices on Regular Trials
SF.MakeFig4(ChoicesCollapsed[TrTypeCollapsed==1],EvidUnitsACollapsed[TrTypeCollapsed==1,:],\
            EvidUnitsBCollapsed[TrTypeCollapsed==1,:],TrErCollapsed[TrTypeCollapsed==1],TrTypeCollapsed[TrTypeCollapsed==1],MethodInput='NarrowBroad')

#Supplementary tables- format the cross-validation results as tables
SF.MakeSupTables(CV_OutputValues,Subjects)

#Code for Figure 8
BootStrapNo = 0 #Bootstrap number to generate error estimates for parameters in the lapsing models. Note, in the paper, this was set to 10000. It has been set to 0 here, which reloads the pre-generated parameters
noPermutations = 0 #Permutation number to compare ketamine and saline parameter estimates for parameters in the lapsing models. Note, in the paper, this was set to 10000. It has been set to 0 here, which reloads the pre-generated parameters

DirToSaveBootstraps = DataFileLocations + '/Bootstrap_Permutation_Data'
DD_FileNames = ['DrugDayProcessedDataSubjectH22-Aug-2020.mat','DrugDayProcessedDataSubjectA12-Feb-2020.mat']
for i in [0,1]:
    #Load the matlab data
    libDD[i] = spio.loadmat(pjoin(DataFileLocations,DD_FileNames[i]))

#Extract key variables
OverallDataStructure = np.append(libDD[0]['DrugDayStructure'],libDD[1]['DrugDayStructure'])
ChosenTarget = np.concatenate(OverallDataStructure['ChosenTarget'],axis=1)
NoSessions = (libDD[0]['DrugDayStructure']['ChosenTarget']).shape[1] + (libDD[1]['DrugDayStructure']['ChosenTarget']).shape[1]

# Analyse the data for each individual session
BinnedAnalysis = np.zeros((NoSessions,81)) #Organise a matrix, to store performance data in each time bin
OutputDMHere=[[0 for j in range(81)]for i in range(NoSessions)] # %Organise a cell array to store design matricies for each time point: Number of Sessions x 81 time points
SessionWiseDMs = [None]*NoSessions

# Loop across sessions
print('Analysing drug day sessions')
for i in range(0,NoSessions):
    SessionWiseDMs[i], BinnedAnalysis[i,:], OutputDMHere[i] = SF.AnalyseWithinSessionDrugDayData(OverallDataStructure[i])

KetamineSes,SalineSes = SF.FindSessionsToInclude(SessionWiseDMs,OverallDataStructure)

SF.MakeFig8(BinnedAnalysis,SessionWiseDMs,KetamineSes,SalineSes,BootStrapNo,noPermutations)
print('Analyses complete')