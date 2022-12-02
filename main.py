import scipy.io as spio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SeanFunc as SF

FilesNames = ['NonDrugDayProcessedData_SubjectH_63_Sessions21-Aug-2020.mat','NonDrugDayProcessedData_SubjectA_41_Sessions21-Aug-2020.mat']
Subjects =['Monkey H','Monkey A']
EvidenceUnitsA_acrossSessions, EvidenceUnitsB_acrossSessions, Choices_acrossSessions,LongSampleTrial, \
    resp_trials,TrialType,DiscardTrials,TrialError,TrialErCompTr,IndexOfTrToUseFromRT,IndexOfTrToUseFromAllTr,CV_OutputValues =\
    [None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2,[None]*2

for i in [0,1]:
    print('Processing data for ' + Subjects[i])

    #Load the matlab data
    lib = spio.loadmat(FilesNames[i])

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
    n_kcv_runs = 5 # How many cross-validation runs to run? Note, in the paper, this was set to 100. It has been set to 5 here to decrease computing time.
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
print('end of code')
