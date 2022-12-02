import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2_contingency, norm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable                                                                 # To make imshow 2D-plots to have color bars at the same height as the figure
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, IndexLocator
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy import stats
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd

#matplotlib settings
#matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend. Will prevent from viewing figs
import matplotlib.cm as matplotlib_cm
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
font = FontProperties(family="Arial")
hfont = {'fontname':'Arial'}
import matplotlib as mpl
mpl.rc('font',family='Arial')
mpl.rcParams['lines.linewidth'] = 1.5
plt.rcParams["font.family"] = "Arial"

#common variables across figures
max1 = 8.5 / 2.54  # involved in calculating figure size
bar_width = 0.8
bar_opacity = 0.75
xbuf0 = 0.11  # x space between figure panels
ybuf0 = 0.08  # y space between figure panels
fontsize_fig_label = 10
fontsize_legend = 8
fontsize_tick = 7
ColorsHere = [(140. / 255, 81. / 255, 10. / 255), (128. / 255, 177. / 255, 211. / 255),
              (102. / 255, 102. / 255, 102. / 255)]
Reg_combined_color_list = [(0.4, 0.7607843137254902, 0.6470588235294118), (0.9137254901960784, 0.6392156862745098, 0.788235294117647), (0.4, 0.7607843137254902, 0.6470588235294118), 'grey', 'grey', 'grey', 'grey']

path_cwd = './'

def MakeFig2(IndexOfTrToUse,EvidenceOptA,EvidenceOptB,Choices,TrialEr,MethodInput,Subject):
    dx_list,P_corr_Subj_list,ErrBar_P_corr_Subj_list,lik_model = \
        PsychometricFit(IndexOfTrToUse, EvidenceOptA, EvidenceOptB, Choices, TrialEr, MethodInput)

    x_values = 100.*dx_list[:-1]

#calculate the weighting of evidence over time
    dm = np.concatenate(
        [np.ones([np.sum(IndexOfTrToUse), 1]), EvidenceOptA[IndexOfTrToUse, :] - EvidenceOptB[IndexOfTrToUse, :]],
        axis=1) #create design matrix for logistic regression
    m = sm.Logit(Choices[IndexOfTrToUse],dm).fit()  # Fit the model
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

######## MAKE FIGURES

## Define subfigure domain.
    figsize = (max1,1.*max1)

    #DEFINE POSITIONS OF FIGURE PANELS
    width1_11 = 0.32; width1_12 = width1_11
    width1_21 = width1_11; width1_22 = width1_21
    x1_11 = 0.135; x1_12 = x1_11 + width1_11 + 1.7*xbuf0
    x1_21 = x1_11; x1_22 = x1_12
    height1_11 = 0.3; height1_12 = height1_11
    height1_21= height1_11;  height1_22 = height1_21
    y1_11 = 0.62; y1_12 = y1_11
    y1_21 = y1_11 - height1_21 - 2.35*ybuf0; y1_22 = y1_21

    rect1_11_0 = [x1_11, y1_11, width1_11*0.05, height1_11]
    rect1_11 = [x1_11+width1_11*0.2, y1_11, width1_11*(1-0.2), height1_11]
    rect1_12_0 = [x1_12, y1_12, width1_12*0.05, height1_12]
    rect1_12 = [x1_12+width1_12*0.2, y1_12, width1_12*(1-0.2), height1_12]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]
    if Subject=='Monkey H':
        PKPosBox = rect1_22
        PsychBox = rect1_12
        PsychBox_subfig = rect1_12_0

        ##### Set up the panel labels
        fig_temp = plt.figure(num=2, figsize=figsize)
        fig_temp.text(0.01, 0.915, 'A', fontsize=fontsize_fig_label, fontweight='bold')
        fig_temp.text(0.015 + x1_12 - x1_11, 0.915, 'B', fontsize=fontsize_fig_label, fontweight='bold')
        fig_temp.text(0.01, 0.915 + y1_22 - y1_12, 'C', fontsize=fontsize_fig_label, fontweight='bold')
        fig_temp.text(0.015 + x1_22 - x1_21, 0.915 + y1_22 - y1_12, 'D', fontsize=fontsize_fig_label, fontweight='bold')
        fig_temp.text(0.185, 0.96, 'Monkey A', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal',
                      color='k')
        fig_temp.text(0.695, 0.96, 'Monkey H', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal',
                      color='k')
    else:
        fig_temp = plt.figure(num=2)
        PKPosBox = rect1_21
        PsychBox = rect1_11
        PsychBox_subfig = rect1_11_0

    ## rect1_12: Psychometric function (over dx_corr), monkey H
    ax_0   = fig_temp.add_axes(PsychBox_subfig) #create a mini axis so the first datapoint is separate
    ax   = fig_temp.add_axes(PsychBox) #create the main axis for the figure
    remove_topright_spines(ax_0)  #remove top line from figure
    remove_topright_spines(ax)  #remove top line from figure
    ax.spines['left'].set_visible(False)
    remove_topright_spines(ax) #remove top line from figure
    # Log-Spaced
    ax.errorbar(x_values[2:],P_corr_Subj_list[2:-1], ErrBar_P_corr_Subj_list[0,2:-1], color='k', markerfacecolor='grey', ecolor='grey', fmt='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
    tmp = ax_0.errorbar(x_values[1], P_corr_Subj_list[1], ErrBar_P_corr_Subj_list[0,1], color='k', markerfacecolor='grey', ecolor='grey', marker='.', zorder=4, clip_on=False                         , markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)#, linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    for b in tmp[2]:
        b.set_clip_on(False)
    x_list_psychometric = np.arange(1, 50, 1)
    ax.plot(x_list_psychometric,PsychFitter(x_list_psychometric/100,lik_model['x']) , color='k', ls='-', clip_on=False, zorder=2)
    ax_0.scatter(0, PsychFitter(0,lik_model['x']), s=15., color='k', marker='_', clip_on=False, zorder=2, linewidth=1.305)
    ax.set_xscale('log')
    ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.) #where to centre the XLabel, and distance from axis
    ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-3.) #put the ylabel on the smaller axis
    ax_0.set_ylim([0.48,1.])
    ax.set_ylim([0.48,1.])
    ax_0.set_xlim([-1,1])
    ax.set_xlim([1,50])
    ax_0.set_xticks([0.])
    ax.xaxis.set_ticks([1, 10])
    ax_0.set_yticks([0.5, 1.])
    ax_0.yaxis.set_ticklabels([0.5, 1])

    minorLocator = MultipleLocator(0.1)
    ax_0.yaxis.set_minor_locator(minorLocator)
    ax.set_yticks([]) #remove ticks from the main axis
    ax_0.tick_params(direction='out', pad=1.5)
    ax_0.tick_params(which='minor', direction='out')
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')

    #set a break line between the two axes
    kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
    y_shift_spines = -0.0968
    ax_0.plot((1, 1 + 2. / 3.), (y_shift_spines + 0., y_shift_spines + 0.05), **kwargs)  # top-left diagonal
    ax_0.plot((1 + 2. / 3., 1 + 4. / 3,), (y_shift_spines + 0.05, y_shift_spines - 0.05), **kwargs)  # top-left diagonal
    ax_0.plot((1 + 4. / 3., 1 + 6. / 3.), (y_shift_spines - 0.05, y_shift_spines + 0.), **kwargs)  # top-left diagonal
    ax_0.plot((1 + 6. / 3., 1 + 9. / 3.), (y_shift_spines + 0., y_shift_spines + 0.), **kwargs)  # top-left diagonal
    ax_0.spines['left'].set_position(('outward', 5))
    ax_0.spines['bottom'].set_position(('outward', 7))
    ax.spines['bottom'].set_position(('outward', 7))

    ## PK subpanel
    ax = fig_temp.add_axes(PKPosBox)
    remove_topright_spines(ax)
    ax.errorbar(np.linspace(1,8,8), betas[1:], stdErs[1:], color='k', markerfacecolor='grey', ecolor='grey',
                linestyle='-', marker='.', zorder=(3 - 1), clip_on=False, alpha=1., elinewidth=0.6, markeredgewidth=0.6,
                capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    ax.set_xlabel('Sample Number', fontsize=fontsize_legend, labelpad=1.)
    ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend, labelpad=2.)
    ax.set_ylim([0., 4.05])
    ax.set_xlim([1., 8.])
    ax.set_xticks([1., 8.])
    ax.set_yticks([0., 4.])
    ax.text(0.1, 4.2, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
    minorLocator = MultipleLocator(1.)
    ax.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(1.)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))

    if Subject=='Monkey A': #both subjects habe been processed, figure ready for saving
        fig_temp.savefig(path_cwd+'Figure2_2022.pdf')    #Finally save fig

def remove_topright_spines(ax):
    # hide the top and right spines
    [spin.set_visible(False) for spin in (ax.spines['top'], ax.spines['right'])]

    #hide the right and top tick marks
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    return None

def GetTrialDistributionData(EvidA,EvidB,TrialType,dx_distribution):
    x_distribution_list = np.arange(0, 1 + dx_distribution, dx_distribution)

    n_distribution_regularTr = np.zeros((len(x_distribution_list), len(x_distribution_list)))
    n_distribution_narrow_high = np.zeros((len(x_distribution_list), len(x_distribution_list)))
    n_distribution_broad_high = np.zeros((len(x_distribution_list), len(x_distribution_list)))
    n_distribution_NB_balanced = np.zeros((len(x_distribution_list), len(x_distribution_list)))

    dx_SD_distribution = 0.01;  # Width of the bins used.
    x_SD_distribution_list = np.arange(0, 0.3 + dx_SD_distribution, dx_SD_distribution)  # List of bins used
    n_SD_distribution_regularTr = np.zeros((len(x_SD_distribution_list), len(x_SD_distribution_list)))

    MeanEvidA = np.mean(EvidA,axis=1)
    MeanEvidB = np.mean(EvidB,axis=1)
    MeanEvidDiff = np.mean(EvidA,axis=1)-np.mean(EvidB,axis=1)
    SdEvidA = np.std(EvidA, axis=1,ddof=1)
    SdEvidB = np.std(EvidB, axis=1,ddof=1)
    SdEvidDiff = np.std(EvidA, axis=1,ddof=1) - np.std(EvidB, axis=1,ddof=1)

    for i in range(0,len(SdEvidDiff)):
        if TrialType[i]==1:
            if SdEvidA[i] >= SdEvidB[i]:
                n_distribution_regularTr[np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1,
                                         np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1] =                 \
                    n_distribution_regularTr[np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1,
                                         np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1] + 1;

                n_SD_distribution_regularTr[np.ceil(SdEvidB[i]/dx_SD_distribution).astype(int)-1,np.ceil(SdEvidA[i]/dx_SD_distribution).astype(int)-1] = \
                    n_SD_distribution_regularTr[np.ceil(SdEvidB[i]/dx_SD_distribution).astype(int)-1,np.ceil(SdEvidA[i]/dx_SD_distribution).astype(int)-1] +1

            else:
                n_distribution_regularTr[np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1,
                                         np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1] =                 \
                    n_distribution_regularTr[np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1,
                                         np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1] + 1;

                n_SD_distribution_regularTr[np.ceil(SdEvidA[i]/dx_SD_distribution).astype(int)-1,np.ceil(SdEvidB[i]/dx_SD_distribution).astype(int)-1] = \
                    n_SD_distribution_regularTr[np.ceil(SdEvidA[i]/dx_SD_distribution).astype(int)-1,np.ceil(SdEvidB[i]/dx_SD_distribution).astype(int)-1] +1

        elif TrialType[i]==18:
            n_distribution_narrow_high[np.ceil(MeanEvidB[i]/dx_distribution).astype(int)-1,np.ceil(MeanEvidA[i]/dx_distribution).astype(int)-1] = \
                n_distribution_narrow_high[np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1,
                                          np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1] +1
        elif TrialType[i]==21:
            n_distribution_narrow_high[np.ceil(MeanEvidA[i]/dx_distribution).astype(int)-1,np.ceil(MeanEvidB[i]/dx_distribution).astype(int)-1] = \
                n_distribution_narrow_high[np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1,
                                          np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1] +1
        elif TrialType[i]==19:
            n_distribution_broad_high[np.ceil(MeanEvidB[i]/dx_distribution).astype(int)-1,np.ceil(MeanEvidA[i]/dx_distribution).astype(int)-1]= \
                n_distribution_broad_high[np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1,
                                          np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1] +1
        elif TrialType[i]==22:
            n_distribution_broad_high[np.ceil(MeanEvidA[i]/dx_distribution).astype(int)-1,np.ceil(MeanEvidB[i]/dx_distribution).astype(int)-1] = \
                n_distribution_broad_high[np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1,
                                          np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1] +1
        elif TrialType[i]==20:
            n_distribution_NB_balanced[np.ceil(MeanEvidB[i]/dx_distribution).astype(int)-1,np.ceil(MeanEvidA[i]/dx_distribution).astype(int)-1] = \
                n_distribution_NB_balanced[np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1,
                                          np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1] +1
        elif TrialType[i]==23:
            n_distribution_NB_balanced[np.ceil(MeanEvidA[i]/dx_distribution).astype(int)-1,np.ceil(MeanEvidB[i]/dx_distribution).astype(int)-1] = \
                n_distribution_NB_balanced[np.ceil(MeanEvidA[i] / dx_distribution).astype(int)-1,
                                          np.ceil(MeanEvidB[i] / dx_distribution).astype(int)-1] +1

    return n_distribution_regularTr,n_distribution_narrow_high,n_distribution_broad_high,n_distribution_NB_balanced, n_SD_distribution_regularTr


def Analyse_NarBroad(ChosenTarget,TrialType,EvidA,EvidB):
    #Define the trial types
    BroadLow_Versus_NarrowHigh = TrialType == 18;
    BroadHigh_Versus_NarrowLow = TrialType == 19;
    BroadBalanced_Versus_NarrowBalanced = TrialType == 20;
    NarrowHigh_Versus_BroadLow = TrialType == 21;
    NarrowLow_Versus_BroadHigh = TrialType == 22;
    NarrowBalanced_Versus_BroadBalanced = TrialType == 23;

    NoTr = [(np.sum(BroadLow_Versus_NarrowHigh) + np.sum(NarrowHigh_Versus_BroadLow)),\
    (np.sum(NarrowLow_Versus_BroadHigh) + np.sum(BroadHigh_Versus_NarrowLow)), \
        (np.sum(NarrowBalanced_Versus_BroadBalanced) + np.sum(BroadBalanced_Versus_NarrowBalanced))]

    #Calculate the accuracy and standard error on each of the two trial types
    Accuracy = np.zeros((2,1))
    StdErs = np.zeros((3,1))

    Accuracy[0] = (np.sum(ChosenTarget[BroadLow_Versus_NarrowHigh]==0)+np.sum(ChosenTarget[NarrowHigh_Versus_BroadLow]==1))/NoTr[0]
    Accuracy[1] = (np.sum(ChosenTarget[NarrowLow_Versus_BroadHigh] == 0) + np.sum(
        ChosenTarget[BroadHigh_Versus_NarrowLow] == 1)) /NoTr[1]
    StdErs[0:2] = np.sqrt(((Accuracy*(1-Accuracy)).T)/NoTr[:-1]).T

    #Calculate broad preference and standard error on balanced trials
    BroadPref = (np.sum(ChosenTarget[NarrowBalanced_Versus_BroadBalanced] == 0) + np.sum(
        ChosenTarget[BroadBalanced_Versus_NarrowBalanced] == 1)) /NoTr[2]
    StdErs[2] = np.sqrt(((BroadPref * (1 - BroadPref)).T) / NoTr[-1:]).T

    #Compare accuracy on NarrowHigh and BroadHigh trials using Chi-2 test
    ChiInputRow1 = [(np.sum(ChosenTarget[BroadLow_Versus_NarrowHigh] == 0) + np.sum(
        ChosenTarget[NarrowHigh_Versus_BroadLow] == 1)), (
                             np.sum(ChosenTarget[NarrowLow_Versus_BroadHigh] == 0) + np.sum(
                         ChosenTarget[BroadHigh_Versus_NarrowLow] == 1))]

    ChiInput = [ChiInputRow1,(np.array(NoTr[0:2])-ChiInputRow1).tolist()]
    stat, p, dof, expected = chi2_contingency(ChiInput, correction=False)

    #Compare broad preference on ambigous trials with chance using binomial test
    BinomP = stats.binom_test((np.sum(ChosenTarget[NarrowBalanced_Versus_BroadBalanced] == 0) + np.sum(
        ChosenTarget[BroadBalanced_Versus_NarrowBalanced] == 1)), NoTr[2], p=0.5, alternative='two-sided')

    #visualise information about the trial difficulty
    dx_distribution = 0.0005
    distRegTr,distNarHighTr,distBroadHighTr,distBalancedTr,distSdRegTr\
        = GetTrialDistributionData(EvidA,EvidB,TrialType,dx_distribution)

    density_distribution_narrow_high_all = (distNarHighTr) / np.sum(distNarHighTr)
    density_distribution_broad_high_all = (distBroadHighTr) / np.sum(distBroadHighTr)
    density_distribution_NB_balanced_all = (distBalancedTr) / np.sum(distBalancedTr)
    density_distribution_net_narrow_high_all = np.zeros(len(density_distribution_narrow_high_all))
    density_distribution_net_broad_high_all = np.zeros(len(density_distribution_broad_high_all))
    density_distribution_net_NB_balanced_all = np.zeros(len(density_distribution_NB_balanced_all))

    #get a distribution for NET evidence (i.e. 1D from the 2D matrix)
    for i in range(len(density_distribution_net_narrow_high_all)):
        density_distribution_net_narrow_high_all[i] = np.sum(density_distribution_narrow_high_all.diagonal(
            i - int((len(density_distribution_narrow_high_all) - 1.) / 2.)))
        density_distribution_net_broad_high_all[i] = np.sum(
            density_distribution_broad_high_all.diagonal(i - int((len(density_distribution_broad_high_all) - 1.) / 2.)))
        density_distribution_net_NB_balanced_all[i] = np.sum(density_distribution_NB_balanced_all.diagonal(
            i - int((len(density_distribution_NB_balanced_all) - 1.) / 2.)))

    #smooth the net evidence distributions
    dx_density = 0.05
    length_density = int(100 / dx_density) + 1
    x_pm = 2
    n_x_smooth = 40

    density_distribution_net_narrow_high_all_smooth_2p = sliding_win_on_lin_data(
        density_distribution_net_narrow_high_all[(int(52 / dx_density) + 1):(int(64 / dx_density) + 2)], n_x_smooth)
    density_distribution_net_narrow_high_all_smooth_n2m = sliding_win_on_lin_data(
        density_distribution_net_narrow_high_all[(int(36 / dx_density) + 1):(int(48 / dx_density) + 2)], n_x_smooth)
    density_distribution_net_narrow_high_all_smooth = np.zeros(length_density)
    density_distribution_net_narrow_high_all_smooth[
    (int(52 / dx_density) + 1):(int(64 / dx_density) + 2)] = density_distribution_net_narrow_high_all_smooth_2p
    density_distribution_net_narrow_high_all_smooth[
    (int(36 / dx_density) + 1):(int(48 / dx_density) + 2)] = density_distribution_net_narrow_high_all_smooth_n2m

    density_distribution_net_broad_high_all_smooth_2p = sliding_win_on_lin_data(
        density_distribution_net_broad_high_all[(int(52 / dx_density) + 1):(int(64 / dx_density) + 2)], n_x_smooth)
    density_distribution_net_broad_high_all_smooth_n2m = sliding_win_on_lin_data(
        density_distribution_net_broad_high_all[(int(36 / dx_density) + 1):(int(48 / dx_density) + 2)], n_x_smooth)
    density_distribution_net_broad_high_all_smooth = np.zeros(length_density)
    density_distribution_net_broad_high_all_smooth[
    (int(52 / dx_density) + 1):(int(64 / dx_density) + 2)] = density_distribution_net_broad_high_all_smooth_2p
    density_distribution_net_broad_high_all_smooth[
    (int(36 / dx_density) + 1):(int(48 / dx_density) + 2)] = density_distribution_net_broad_high_all_smooth_n2m

    density_distribution_net_NB_balanced_all_smooth_pn4 = sliding_win_on_lin_data(
        density_distribution_net_NB_balanced_all[(int(46 / dx_density) + 1):(int(54 / dx_density) + 2)], n_x_smooth)
    density_distribution_net_NB_balanced_all_smooth = np.zeros(length_density)
    density_distribution_net_NB_balanced_all_smooth[
    (int(46 / dx_density) + 1):(int(54 / dx_density) + 2)] = density_distribution_net_NB_balanced_all_smooth_pn4

    ######## MAKE FIGURES
    ## Define subfigure domain.
    figsize = (max1, 1.6 * max1)

    # ## define the subplot locations - 3 rows
    width1_11 = 0.8;
    width1_22 = 0.185;
    width1_21 = width1_22 * (1 + bar_width) / bar_width;
    width1_31 = width1_21;
    width1_32 = width1_22
    x1_11 = 0.14;
    x1_21 = x1_11 - 0.18 * xbuf0;
    x1_22 = x1_21 + width1_21 + 1.8 * xbuf0;
    x1_31 = x1_21;
    x1_32 = x1_22
    height1_11 = 0.22;
    height1_21 = height1_11;
    height1_22 = height1_21;
    height1_31 = height1_21;
    height1_32 = height1_22
    y1_11 = 0.725;
    y1_21 = y1_11 - height1_21 - 1.5 * ybuf0;
    y1_22 = y1_21;
    y1_31 = y1_21 - height1_31 - 1.25 * ybuf0;
    y1_32 = y1_31

    rect1_11 = [x1_11, y1_11, width1_11, height1_11]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]
    rect1_31 = [x1_31, y1_31, width1_31, height1_31]
    rect1_32 = [x1_32, y1_32, width1_32, height1_32]

    ##### Plot the figure letters and panel titles
    fig_temp = plt.figure(num=3,figsize=figsize)
    fig_temp.text(0.5, 0.97, 'Narrow-Broad Trials', fontsize=fontsize_fig_label, fontweight='bold',
                  rotation='horizontal', color='k', horizontalalignment='center')
    fig_temp.text(0.01, 0.945, 'A', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.5, 0.97 + y1_21 - y1_11, 'Monkey (current experiment)', fontsize=fontsize_fig_label - 1,
                  fontweight='bold', rotation='horizontal', color='k', horizontalalignment='center')
    fig_temp.text(0.01, 0.94 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.01 + x1_22 - x1_21, 0.94 + y1_21 - y1_11, 'C', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.5, 0.97 + y1_31 - y1_11, 'Human (Tsetsos et al., ' + r'$\bf{\it{PNAS}}$' + ' 2012)',
                  fontsize=fontsize_fig_label - 1, fontweight='bold', rotation='horizontal', color='k',
                  horizontalalignment='center')
    fig_temp.text(0.01, 0.94 + y1_31 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.01 + x1_32 - x1_31, 0.94 + y1_31 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')


    #Plot the first panel
    ax = fig_temp.add_axes(rect1_11)
    remove_topright_spines(ax)
    x_schem = np.arange(1, 99, 1)
    sigma_narrow = 12
    sigma_broad = 24

    dist_narrow = norm.pdf(x_schem, 54., sigma_narrow)
    dist_broad = norm.pdf(x_schem, 54. - 8., sigma_broad)
    x_net_list_temp = 100. / len(density_distribution_net_narrow_high_all_smooth) * np.arange(
        -int((len(density_distribution_net_narrow_high_all_smooth) - 1.) / 2.),
        len(density_distribution_net_narrow_high_all_smooth) - int(
            (len(density_distribution_net_narrow_high_all_smooth) - 1.) / 2.))
    ax.plot(x_net_list_temp, density_distribution_net_narrow_high_all_smooth, color=ColorsHere[0], ls='-', clip_on=True,
            zorder=10, label='Narrow Correct')  # , linestyle=linestyle_list[i_var_a])
    ax.plot(x_net_list_temp, density_distribution_net_NB_balanced_all_smooth, color=ColorsHere[2], ls='-', clip_on=True,
            zorder=11, label='Ambiguous')  # , linestyle=linestyle_list[i_var_a])
    ax.plot(x_net_list_temp, density_distribution_net_broad_high_all_smooth, color=ColorsHere[1], ls='-', clip_on=True,
            zorder=10, label='Broad Correct')  # , linestyle=linestyle_list[i_var_a])
    ax.set_xlabel('Evidence Strength (Broad minus Narrow)', fontsize=fontsize_legend, labelpad=1.)
    ax.set_ylabel('Probability Density', fontsize=fontsize_legend, labelpad=2.)
    ax.set_xlim([-15., 15.])
    ax.set_ylim([0., 0.01])
    ax.set_xticks([-15, 0, 15])
    minorLocator = MultipleLocator(5)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.set_yticks([0, 0.01])
    ax.yaxis.set_ticklabels([0, 1])
    minorLocator = MultipleLocator(0.002)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')
    ax.spines['left'].set_position(('outward', 5))
    legend = ax.legend(loc=(0.03, 0.87), fontsize=fontsize_legend - 2, frameon=False, ncol=3, markerscale=0.,
                       columnspacing=1.5, handletextpad=0.)
    for color, text, item in zip([ColorsHere[0], ColorsHere[2], ColorsHere[1]], legend.get_texts(), legend.legendHandles):
        text.set_color(color)
        item.set_visible(False)
    ax.text(-16.5, 0.0105, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick - 1.)

    #Store the key variables to make a for loop across panels with same design
    PanelLocation = np.concatenate((rect1_21,rect1_22,rect1_31,rect1_32)).reshape(2, *(2,4))

    DataFromHumanStudy = np.array([ 0.671, 0.781,0.623,])
    ErFromHumanStudy = np.array([0.065/2., 0.08979/2.,0.09598/2.])

    BarHeights = [np.append(Accuracy,BroadPref),DataFromHumanStudy]
    BarErs = [StdErs,ErFromHumanStudy]

    for ii in [0,1]:
        ##Plot the bar charts - first subplot
        ax = fig_temp.add_axes(PanelLocation[ii,0,:])
        remove_topright_spines(ax)
        ax.bar([0, 1], BarHeights[ii][:-1], bar_width, alpha=bar_opacity, yerr=BarErs[ii][:-1],
               ecolor='k', color=ColorsHere, clip_on=False, align='edge', linewidth=1.,
               error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
        ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(7., 3.5))
        ax.scatter([0.4, 1.4, 0.9], [0.535, 0.535, 1.01], s=16., color='k', marker=(5, 2), clip_on=False,
                   zorder=10)  # , linestyle=linestyle_list[i_var_a])
        ax.plot([0.4, 1.4], [0.96, 0.96], ls='-', lw=1., color='k', clip_on=False,
                zorder=9)  # , linestyle=linestyle_list[i_var_a])
        ax.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
        ax.set_xlim([0, 1 + bar_width])
        ax.set_ylim([0., 1.])
        ax.set_xticks([bar_width / 2. - 0., 1 + bar_width / 2. + 0.])
        ax.xaxis.set_ticklabels(['Narrow Correct', 'Broad Correct'])
        ax.set_yticks([0., 0.5, 1.])
        ax.yaxis.set_ticklabels([0, 0.5, 1])
        minorLocator = MultipleLocator(0.25)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.tick_params(direction='out', pad=1.5)
        ax.tick_params(which='minor', direction='out')
        ax.tick_params(bottom="off")

        ## plot the bar charts - broad preference with equal mean
        ax = fig_temp.add_axes(PanelLocation[ii,1,:])
        remove_topright_spines(ax)
        ax.bar([0], BarHeights[ii][-1], bar_width, alpha=bar_opacity, yerr=BarErs[ii][-1], ecolor='k',
               color=ColorsHere[2], clip_on=False, align='edge', linewidth=1.,
               error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
        ax.axhline(y=0.5, color='k', ls='--', lw=0.9, dashes=(7.5, 3.75))
        ax.scatter(0.4, 0.535, s=16., color='k', marker=(5, 2), clip_on=False,
                   zorder=10)  # , linestyle=linestyle_list[i_var_a])
        ax.set_ylabel('Broad Preference', fontsize=fontsize_legend, labelpad=2.)
        ax.set_xlim([0, bar_width])
        ax.set_ylim([0., 1.])
        ax.set_xticks([bar_width / 2. - 0.])
        ax.xaxis.set_ticklabels(['Ambiguous'])
        ax.set_yticks([0., 0.5, 1.])
        ax.yaxis.set_ticklabels([0, 0.5, 1])
        minorLocator = MultipleLocator(0.25)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.tick_params(direction='out', pad=1.5)
        ax.tick_params(which='minor', direction='out')
        ax.tick_params(bottom="off")

    fig_temp.savefig(path_cwd + 'Figure3_2022.pdf')  # Finally save fig

def sliding_win_on_lin_data(data_mat,window_width,axis=0):
    smaller_half = np.floor((window_width)/2)
    bigger_half = np.ceil((window_width)/2)
    data_mat_result = np.zeros(len(data_mat))
    for k_lin in range(len(data_mat)):
        lower_bound = math.floor(np.maximum(k_lin-smaller_half, 0))
        upper_bound = math.floor(np.minimum(k_lin+bigger_half, len(data_mat)))
        data_mat_result[k_lin] = np.mean(data_mat[lower_bound:upper_bound])
    return data_mat_result

def PsychometricFit(IndexOfTrToUse,EvidenceOptA,EvidenceOptB,Choices,TrialEr,MethodInput):
    # Define some important variables
    nTr = sum(IndexOfTrToUse)
    dx_pos_log_list = np.logspace(np.log(0.02) / np.log(10), np.log(0.4) / np.log(10),
                                  10)  # Initial bin spacing for evidence values                                                       % x for evidence
    BestOption = np.empty([nTr, 1])
    BestOption[np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) > np.mean(
        EvidenceOptB[IndexOfTrToUse, :], axis=1)] = 1
    BestOption[np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) < np.mean(
        EvidenceOptB[IndexOfTrToUse, :], axis=1)] = 2
    BestOption[np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(
        EvidenceOptB[IndexOfTrToUse, :], axis=1)] = 0
    Choices = Choices[IndexOfTrToUse]
    TrialEr = TrialEr[IndexOfTrToUse]

    # Create the bins
    if MethodInput == 'CorrectIncorrect':
        dx_list = np.append([0, 0.01], dx_pos_log_list)
    elif MethodInput == 'NarrowBroad':
        dx_list = np.sort(np.concatenate((dx_pos_log_list, -dx_pos_log_list, np.array([0, 0.01, -0.01])), axis=0))

    P_corr_Subj_list = np.zeros(dx_list.shape)
    n_dx_list = np.zeros(dx_list.shape)

    # Indexed groupings
    if MethodInput == 'CorrectIncorrect':
        idx_pos_log_Collapsed = dx_pos_log_list.shape[0] - np.round((np.log(np.abs(
            np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) - np.mean(EvidenceOptB[IndexOfTrToUse, :],
                                                                       axis=1))) - np.log(dx_list[-1])) \
                                                                    / (np.log(dx_pos_log_list[0]) - np.log(
            dx_pos_log_list[-1])) * (np.shape(dx_pos_log_list)[0] - 1))
        # Log-Spaced. Ignoring signs and only map to positive log-space (from 1 to length(dx_pos_log_list)).

        idx_Collapsed = idx_pos_log_Collapsed + 2;
        idx_Collapsed[
            idx_pos_log_Collapsed < 1] = 2  # 1 in i_dx_pos_log_Collapsed => absolute diff in evidence is less than the minimum for dx_pos_log_list(1)=0.02 (~0.017).
        idx_Collapsed[
            (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))] = 1
    elif MethodInput == 'NarrowBroad':
        is_A_broad = (2*np.heaviside(EvidenceOptA[IndexOfTrToUse,:].std(axis=1)-EvidenceOptB[IndexOfTrToUse,:].std(axis=1),0.5))-1
        is_A_broad[is_A_broad==0] = 1

        i_dx_pos_log_Collapsed_NB = len(dx_pos_log_list) - np.round( \
            (np.log(np.abs(EvidenceOptA[IndexOfTrToUse,:].mean(axis=1)-\
                                                  EvidenceOptB[IndexOfTrToUse,:].mean(axis=1)))\
                                    -np.log(dx_pos_log_list[-1])) \
                                    / (np.log(dx_pos_log_list[0]) - np.log(dx_pos_log_list[-1]))* (len(dx_pos_log_list) - 1))

        idx_Collapsed = i_dx_pos_log_Collapsed_NB + 1
        idx_Collapsed[i_dx_pos_log_Collapsed_NB < 1] = 1
        idx_Collapsed[np.mean(EvidenceOptA[IndexOfTrToUse,:], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse,:], axis=1)] = 0
        idx_Collapsed = np.sign([np.mean(EvidenceOptA[IndexOfTrToUse,:], axis=1) - np.mean(EvidenceOptB[IndexOfTrToUse,:], axis=1)])*idx_Collapsed
        idx_Collapsed = idx_Collapsed*is_A_broad + len(dx_pos_log_list) + 2;
        idx_Collapsed= idx_Collapsed.reshape(-1)
        # Setting up the Y-variable
    if MethodInput == 'CorrectIncorrect':  # Y variable is whether the correct option was chosen
        ChoseCorrect = (BestOption.reshape(-1) == 1 * Choices) + (BestOption.reshape(-1) == 2 * (Choices == 0))
        ChoseCorrect[
            (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))] = \
        TrialEr[
            (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))] == 0
        YVar = ChoseCorrect
    elif MethodInput == 'NarrowBroad':
        Broad_Option_Chosen_Collapsed = 0.5 * (is_A_broad + 1) * (Choices) \
        + -0.5 * (is_A_broad - 1)* (1 - (Choices))
        YVar = Broad_Option_Chosen_Collapsed

    # For loop across bins
    for i in range(1, np.shape(dx_list)[0]):
        P_corr_Subj_list[i - 1] = np.sum(YVar[idx_Collapsed == i]) / np.sum(idx_Collapsed == i)
        n_dx_list[i - 1] = np.sum(idx_Collapsed == i)

    # Calculate errorbars
    ErrBar_P_corr_Subj_list = np.sqrt(P_corr_Subj_list * [1 - P_corr_Subj_list] / n_dx_list)

    if MethodInput == 'CorrectIncorrect':
        # fit the psychometric actual function
        evidence_list = np.abs(
        np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) - np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))

        def MleFitSC(params):
            a = params[0]
            b = params[1]
            preds = 0.5 + 0.5 * (1 - np.exp(-np.power((evidence_list / a), b)))
            return -np.sum(np.log((YVar * preds) + ((1 - YVar) * (1 - preds))))

        initParams = [0.05, 1]
        lik_model = minimize(MleFitSC, initParams, method='L-BFGS-B')

    elif MethodInput == 'NarrowBroad':
        # fit the psychometric actual function
        evidence_list = (
            np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) - np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))* is_A_broad

        def MleFitSC_NB(params):
            a = params[0]
            b = params[1]
            c = params[2]

            preds = 0.5 + 0.5 * np.sign(evidence_list + c)* (1. - np.exp(
                -np.power((np.abs(evidence_list + c) / a),b)))

            return -np.sum(np.log((YVar * preds) + ((1 - YVar) * (1 - preds))))

        initParams = [0.05, 1,0]
        lik_model = minimize(MleFitSC_NB, initParams, method='L-BFGS-B')

    return dx_list,P_corr_Subj_list,ErrBar_P_corr_Subj_list,lik_model

def PsychFitter(evidence_list, params):
    a = params[0]
    b = params[1]
    return 0.5 + 0.5 * (1 - np.exp(-np.power((evidence_list / a), b)))

def PsychFitterNB(evidence_list, params):
    a = params[0]
    b = params[1]
    c = params[2]
    return 0.5 + 0.5 * np.sign(evidence_list + c)* (1. - np.exp(
                -np.power((np.abs(evidence_list + c) / a),b)))

def MakeFig4(Choices,EvidenceOptA,EvidenceOptB,TrialEr,TrialType,MethodInput):
    #Fit the psychometric function split by higher SD and lower SD option correct

    dx_list,P_corr_Subj_list,ErrBar_P_corr_Subj_list,lik_model = \
        PsychometricFit(np.ones(len(Choices))>0, EvidenceOptA, EvidenceOptB, Choices, TrialEr, MethodInput)

    #Fit a logistic regression model with Mean Evidence difference and Std Evidence difference as regressors

    Regressor1 = ((np.mean(EvidenceOptA, axis=1) - np.mean(EvidenceOptB, axis=1)).reshape(len(EvidenceOptB), 1))
    Regressor2 = ((np.std(EvidenceOptA, axis=1,ddof=1) - np.std(EvidenceOptB, axis=1,ddof=1)).reshape(len(EvidenceOptB), 1))
    dm = np.concatenate((np.ones((len(EvidenceOptA), 1)),Regressor1,Regressor2), axis = 1)

    m = sm.Logit(Choices==1,dm).fit()
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    ## Extract number distribution of stimuli, for Standard/Regression trials.
    dx_Reg_density = 0.1
    n_x_Reg_smooth = 20

    dx_distribution = 0.02;  # Width of the bins used.
    distRegTr, distNarHighTr, distBroadHighTr, distBalancedTr, distSdRegTr \
        = GetTrialDistributionData(EvidenceOptA, EvidenceOptB, TrialType,dx_distribution)

    density_distribution_Regression_all = distRegTr / np.sum(
        distRegTr)
    density_distribution_net_Regression_all = np.zeros(len(density_distribution_Regression_all))
    for i in range(len(density_distribution_net_Regression_all)):
        density_distribution_net_Regression_all[i] = np.sum(
            density_distribution_Regression_all.diagonal(i - int((len(density_distribution_Regression_all) - 1.) / 2.)))
    density_distribution_net_Regression_all_smooth = sliding_win_on_lin_data(
        density_distribution_net_Regression_all, n_x_Reg_smooth)

    density_SD_distribution_Regression_all = distSdRegTr / np.sum(distSdRegTr)
    density_SD_distribution_net_Regression_all = np.zeros((2 * len(density_SD_distribution_Regression_all) - 1))
    for i in range(2 * len(density_SD_distribution_Regression_all) - 1):
        density_SD_distribution_net_Regression_all[i] = np.sum(density_SD_distribution_Regression_all.diagonal(
            i - int((len(density_SD_distribution_Regression_all) - 1.))))
    density_SD_distribution_net_Regression_all_smooth_0_20 = sliding_win_on_lin_data(
        density_SD_distribution_net_Regression_all[(int(30 / dx_Reg_density) + 1):(int(50 / dx_Reg_density) + 2)],
        n_x_Reg_smooth)
    density_SD_distribution_net_Regression_all_smooth = np.zeros(len(density_SD_distribution_net_Regression_all))
    density_SD_distribution_net_Regression_all_smooth[(int(30 / dx_Reg_density) + 1):(
                int(50 / dx_Reg_density) + 2)] = density_SD_distribution_net_Regression_all_smooth_0_20

    #Make the figures
    x_list_psychometric = np.arange(0.01, 0.5, 0.01)
    x0_psychometric = 0
    ## Define subfigure domain.
    figsize = (max1, 1.2 * max1)

    width1_11 = 0.3; width1_12 = 0.25; width1_21 = 0.3; width1_22 = 0.25
    x1_11 = 0.15; x1_12 = x1_11 + width1_12 + 2.6 * xbuf0; x1_21 = x1_11; x1_22 = x1_12 - 0.021
    height1_11 = 0.3; height1_12 = 0.24; height1_21 = height1_11; height1_22 = 0.28
    y1_11 = 0.59; y1_12 = y1_11 + 0.038; y1_21 = y1_11 - height1_21 - 2.4 * ybuf0; y1_22 = y1_21 + 0.013

    rect1_11 = [x1_11, y1_11, width1_11, height1_11]
    rect1_12_0 = [x1_12, y1_12, width1_12 * 0.05, height1_12]
    rect1_12 = [x1_12 + width1_12 * 0.2, y1_12, width1_12 * (1 - 0.2), height1_12]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]

    ##### Plot the figure labels
    fig_temp = plt.figure(num=4,figsize=figsize)
    fig_temp.text(0.025, 0.9, 'A', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.5, 0.965, 'Regular Trials', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal',
                  color='k', horizontalalignment='center')
    fig_temp.text(0.025 + x1_12 - x1_11, 0.9, 'C', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.025, 0.9 + y1_21 - y1_11, 'B', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.05 + x1_22 - x1_21, 0.9 + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')

    ### Distribution of stimuli conditions
    ## rect1_11: Stimuli Distribution for narrow-high trials.
    ax = fig_temp.add_axes(rect1_11)
    aspect_ratio = 1.
    plt.imshow(density_distribution_Regression_all, extent=(0., 100., 0., 100.), interpolation='nearest', cmap='BuPu',
               aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_distribution_Regression_all))
    ax.plot([0, 100], [0, 100], color='k', alpha=0.8, ls='--', lw=1.)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([25., 50., 75.])
    ax.set_yticks([25., 50., 75.])
    ax.set_xlim([25., 75.])
    ax.set_ylim([25., 75.])
    ax.tick_params(direction='out', pad=0.75)
    ax.set_xlabel('Mean Evidence\n(Higher SD)', fontsize=fontsize_legend, labelpad=2.)
    ax.set_ylabel('Mean Evidence\n(Lower SD)', fontsize=fontsize_legend, labelpad=1.)
    divider = make_axes_locatable(ax)
    cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
    cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
    cb_temp.set_ticklabels((0, 1, 2))
    cb_temp.ax.xaxis.set_tick_params(pad=1.)
    cax_scale_bar_size.xaxis.set_ticks_position("top")
    ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
    ax.text(76.5, 77.8, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick - 1.)

    ### Distribution of stimuli conditions
    ## rect1_21: Stimuli Distribution for narrow-high trials.
    ax = fig_temp.add_axes(rect1_21)
    aspect_ratio = 1.
    plt.imshow(density_SD_distribution_Regression_all, extent=(0., 30., 0., 30.), interpolation='nearest', cmap='BuPu',
               aspect=aspect_ratio, origin='lower', vmin=0., vmax=np.max(density_SD_distribution_Regression_all))
    ax.plot([0, 30], [0, 30], color='k', alpha=0.8, ls='--', lw=1.)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([12., 24.])
    ax.set_yticks([12., 24.])
    ax.set_xlim([8., 28.])
    ax.set_ylim([8., 28.])
    ax.tick_params(direction='out', pad=0.75)
    ax.set_xlabel('Evidence SD\n(Higher SD)', fontsize=fontsize_legend, labelpad=2.)
    ax.set_ylabel('Evidence SD\n(Lower SD)', fontsize=fontsize_legend, labelpad=0.)
    divider = make_axes_locatable(ax)
    cax_scale_bar_size = divider.append_axes("top", size="5%", pad=0.05)
    cb_temp = plt.colorbar(ticks=[0., 0.01, 0.02], cax=cax_scale_bar_size, orientation='horizontal')
    cb_temp.set_ticklabels((0, 1, 2))
    cb_temp.ax.xaxis.set_tick_params(pad=1.)
    cax_scale_bar_size.xaxis.set_ticks_position("top")
    ax.set_title("Trial Frequency", fontsize=fontsize_legend, x=0.49, y=1.2)
    ax.text(28.5, 29., r'$\times \mathregular{10^{-3}}$', fontsize=fontsize_tick - 1.)

    ## rect1_12: Psychometric function (over dx_broad, or dx_corr ?), Monkey A
    ax_0 = fig_temp.add_axes(rect1_12_0)
    ax = fig_temp.add_axes(rect1_12)
    remove_topright_spines(ax_0)
    remove_topright_spines(ax)
    ax.spines['left'].set_visible(False)
    remove_topright_spines(ax)
    # Log-Spaced
    d_evidence_avg_list = 100 * dx_list[1:-1]
    P_corr_avg_list = P_corr_Subj_list[1:-1]
    ErrBar_P_corr_avg_list = ErrBar_P_corr_Subj_list[0][1:-1]
    ax.errorbar(d_evidence_avg_list[12:], P_corr_avg_list[12:], ErrBar_P_corr_avg_list[12:], color=ColorsHere[1],
                ecolor=ColorsHere[1], fmt='.', zorder=4, clip_on=False, label='Higher SD Correct', markeredgecolor='k',
                linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    ax.errorbar(-d_evidence_avg_list[1:9], 1. - P_corr_avg_list[1:9], ErrBar_P_corr_avg_list[1:9], color=ColorsHere[0],
                ecolor=ColorsHere[0], fmt='.', zorder=3, clip_on=False, label='Lower SD Correct', markeredgecolor='k',
                linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    tmp = ax_0.errorbar(d_evidence_avg_list[11], P_corr_avg_list[11], ErrBar_P_corr_avg_list[11], color=ColorsHere[1],
                        ecolor=ColorsHere[1], marker='.', zorder=4, clip_on=False, markeredgecolor='k', linewidth=0.3,
                        elinewidth=0.6, markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    for b in tmp[2]:
        b.set_clip_on(False)
    tmp = ax_0.errorbar(-d_evidence_avg_list[9], 1. - P_corr_avg_list[9], ErrBar_P_corr_avg_list[9], color=ColorsHere[0],
                        ecolor=ColorsHere[0], marker='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3,
                        elinewidth=0.6, markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    for b in tmp[2]:
        b.set_clip_on(False)
    ax.plot(100. * x_list_psychometric, PsychFitterNB(x_list_psychometric,lik_model.x),
            color=ColorsHere[1], ls='-', clip_on=False)  # , linestyle=linestyle_list[i_var_a])
    ax.plot(100. * x_list_psychometric,
            1. - PsychFitterNB(-x_list_psychometric,lik_model.x), color=ColorsHere[0],
            ls='-', clip_on=False)  # , linestyle=linestyle_list[i_var_a])
    ax_0.scatter(100. * x0_psychometric, PsychFitterNB(x0_psychometric,lik_model.x),
                 s=15., color=ColorsHere[1], marker='_', clip_on=False,
                 linewidth=1.305)  # , linestyle=linestyle_list[i_var_a])
    ax_0.scatter(100. * x0_psychometric,
                 1. - PsychFitterNB(-x0_psychometric,lik_model.x), s=15.,
                 color=ColorsHere[0], marker='_', clip_on=False,
                 linewidth=1.305)  # , linestyle=linestyle_list[i_var_a])ax.plot([0.003, 0.5], [0.5,0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
    ax.plot([0.3, 50], [0.5, 0.5], linewidth=0.7, color='k', ls='--', clip_on=False, zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4, labelpad=1.)
    ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=2.)
    ax_0.set_ylim([0.4, 1.])
    ax.set_ylim([0.4, 1.])
    ax_0.set_xlim([-1, 1])
    ax.set_xlim([1, 50])
    ax_0.set_xticks([0.])
    ax.xaxis.set_ticks([1, 10])
    ax_0.set_yticks([0.5, 1.])
    ax_0.yaxis.set_ticklabels([0.5, 1])
    minorLocator = MultipleLocator(0.1)
    ax_0.yaxis.set_minor_locator(minorLocator)
    ax.set_yticks([])
    ax_0.tick_params(direction='out', pad=2.7)
    ax_0.tick_params(which='minor', direction='out')
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')
    ## Add breakmark = wiggle
    kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
    y_shift_spines = -0.1009
    ax_0.plot((1, 1 + 2. / 3.), (y_shift_spines + 0., y_shift_spines + 0.05), **kwargs)  # top-left diagonal
    ax_0.plot((1 + 2. / 3., 1 + 4. / 3,), (y_shift_spines + 0.05, y_shift_spines - 0.05), **kwargs)  # top-left diagonal
    ax_0.plot((1 + 4. / 3., 1 + 6. / 3.), (y_shift_spines - 0.05, y_shift_spines + 0.), **kwargs)  # top-left diagonal
    ax_0.plot((1 + 6. / 3., 1 + 9. / 3.), (y_shift_spines + 0., y_shift_spines + 0.), **kwargs)  # top-left diagonal
    ax_0.spines['left'].set_position(('outward', 5))
    ax_0.spines['bottom'].set_position(('outward', 7))
    ax.spines['bottom'].set_position(('outward', 7))
    legend_bars = [Line2D([0], [0], color=ColorsHere[1], alpha=1., label='Higher SD Correct'),
                   Line2D([0], [0], color=ColorsHere[0], alpha=1., label='Lower SD Correct')]
    legend = ax.legend(handles=legend_bars, loc=(-0.35, -0.12), fontsize=fontsize_legend - 2, frameon=False, ncol=1,
                       markerscale=0., columnspacing=0.5, handletextpad=0., labelspacing=0.3)
    for color, text, item in zip([ColorsHere[1], ColorsHere[0]], legend.get_texts(), legend.legendHandles):
        text.set_color(color)
        item.set_visible(False)

    ### Mean and Var only. L/R difference
    ax = fig_temp.add_axes(rect1_22)
    remove_topright_spines(ax)
    ax.bar(np.arange(len(betas[1:])), betas[1:],
           bar_width, yerr=stdErs[1:], ecolor='k', alpha=1,
           color=Reg_combined_color_list[0:2], clip_on=False, align='edge', linewidth=1.,
           error_kw=dict(elinewidth=0.6, markeredgewidth=0.6), capsize=2.)
    ax.scatter([0.4, 1.4], [22.5, 5.], s=16., color='k', marker=(5, 2), clip_on=False,
               zorder=10)  # , linestyle=linestyle_list[i_var_a])
    ax.set_ylabel('Beta', fontsize=fontsize_legend, labelpad=2.)
    ax.set_xlim([0, len(betas[1:]) - 1 + bar_width])
    ax.set_ylim([0., 23.5])
    ax.set_xticks(np.arange(len(betas[1:])) + bar_width / 2.)
    ax.xaxis.set_ticklabels(['Mean\nEvidence', 'Evidence\nSD'])  # , 'Mean', 'Max', 'Min', 'First', 'Last'])
    ax.set_yticks([0., 20.])
    ax.set_yticklabels([0, 0.2])
    minorLocator = MultipleLocator(5.)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=1.)
    ax.tick_params(which='minor', direction='out')
    ax.tick_params(bottom="off")
    ax.spines['bottom'].set_position(('zero'))

    fig_temp.savefig(path_cwd + 'Figure4_2022.pdf')  # Finally save fig

def AdvancedRegrAnalysis(Choices,EvidA,EvidB,BootStrapNo,n_kcv_runs):

    #Make an initial design matrix
    dm = np.stack((np.ones(len(EvidA)), \
                   EvidA[:, 0], EvidA[:, -1], np.mean(EvidA, axis=1), np.max(EvidA,axis=1), np.min(EvidA,axis=1), \
                   EvidB[:, 0], EvidB[:, -1], np.mean(EvidB, axis=1), np.max(EvidB, axis=1), np.min(EvidB, axis=1))).T
    m = sm.Logit(Choices,dm).fit()
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    #specify the full design matrix, then specify several nested models
    dm_full = np.stack((np.ones(len(EvidA)), \
                   EvidA[:, 0], EvidA[:, -1], np.mean(EvidA, axis=1), np.std(EvidA,axis=1,ddof=1), np.max(EvidA,axis=1), np.min(EvidA,axis=1), \
                   EvidB[:, 0], EvidB[:, -1], np.mean(EvidB, axis=1), np.std(EvidB,axis=1,ddof=1),  np.max(EvidB, axis=1), np.min(EvidB, axis=1))).T

    Listofallmodels = [dm_full[:,(0,1,2,3,5,6,7,8,9,11,12)],   #No SD
                       dm_full, #Full model
                       dm_full[:,(0,1,2,3,4,7,8,9,10)], #No max/min
                       dm_full[:,(0,3,4,5,6,9,10,11,12)], #no first/last
                       dm_full[:,(0,3,4,9,10)], #mean and SD
                       dm_full[:,(0,3,5,6,9,11,12)], #mean max min
                       dm_full[:,(0,1,2,3,7,8,9)], #First last mean
                       dm_full[:, (0, 3, 9)],  # mean only
                       dm_full[:, (0,1,2,4,5,6,7,8,10,11,12)]]  # No mean
    #Set up the cross-validation
    rkf = RepeatedKFold(n_splits=10, n_repeats=n_kcv_runs)
    TotalCost = np.zeros(len(Listofallmodels)); RunsComplete = 0
    for train, test in rkf.split(Choices):
        for ii in range(0,len(Listofallmodels)):
            dm =Listofallmodels[ii]
            m = sm.Logit(Choices[train], dm[train,:]).fit(disp=0)
            TotalCost[ii] = TotalCost[ii] + ((-log_loss(Choices[test], m.predict(dm[test, :])) * len(test)) / rkf.n_repeats)
        RunsComplete = RunsComplete+1
        if np.remainder((RunsComplete/n_kcv_runs*10),10)==0:
            print('Completed ' +  str(RunsComplete/n_kcv_runs*10) + ' % of cross-validation runs')

    #format the output as tables
    OutputValues = [None]*3
    #table 1 = Difference in log-likelihood of Full regression model (mean, SD, max, min, first, last of evidence values; equation 6 in Methods) vs reduced model
    OutputValues[0] = np.array([TotalCost[1]-TotalCost[8],TotalCost[1]-TotalCost[3],TotalCost[1]-TotalCost[0],TotalCost[1]-TotalCost[2]])
    #table 2 = Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max & Min) as regressors, for each monkey and the circuit model
    OutputValues[1] = np.array([TotalCost[4]-TotalCost[5],TotalCost[2]-TotalCost[0]])
    #table 3 = Increase in log-likelihood of various regression models (regressors in column labels) due to inclusion of evidence standard deviation as a regressor
    OutputValues[2] = np.array([TotalCost[4]-TotalCost[7],TotalCost[2]-TotalCost[6],TotalCost[3]-TotalCost[5],TotalCost[1]-TotalCost[0] ])

    return OutputValues

def MakeSupTables(OutputValues,Subjects):
    fig = plt.figure(num=5,figsize=(9, 6))
    DefFontSize = 8
    rectPos = [[0.1,0.65,0.8,0.2],
               [0.1, 0.35, 0.8, 0.2],
               [0.1, 0.05, 0.8, 0.2]]

    df = pd.DataFrame(np.concatenate((OutputValues[0][0], OutputValues[1][0])).reshape(2, 4),
                      columns=['Mean', 'First/Last', 'SD', 'Max/min'], index=Subjects).round(1)
    print(df)
    ax = fig.add_axes(rectPos[0])
    table = ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc="center left")
    table.auto_set_font_size(False)
    table.set_fontsize(DefFontSize)
    ax.set_title("Supplementary Table 1: Difference in log-likelihood of Full regression model (mean, SD, max, min, first, last of evidence values; Equation 6 in Materials and methods) vs reduced model, for each monkey", loc='center', wrap=True,fontweight='bold')
    ax.axis("off")

    df = pd.DataFrame(np.concatenate((OutputValues[0][1], OutputValues[1][1])).reshape(2, 2),
                      columns=['Mean', 'Mean & First & Last'], index=Subjects).round(1)
    print(df)
    ax = fig.add_axes(rectPos[1])
    table = ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc="center left",
             colWidths=[0.25,0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(DefFontSize)
    ax.set_title("Supplementary Table 2: Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max and Min) as regressors, for each monkey", loc='center', wrap=True,fontweight='bold')
    ax.axis("off")

    df = pd.DataFrame(np.concatenate((OutputValues[0][2], OutputValues[1][2])).reshape(2, 4),
                      columns=['Mean', 'Mean, First & Last', 'Mean, Max & Min', 'Mean, Max, Min, First & Last'], index=Subjects).round(1)
    print(df)
    ax = fig.add_axes(rectPos[2])
    table = ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc="center left")
    table.auto_set_font_size(False)
    table.set_fontsize(DefFontSize)
    ax.set_title("Supplementary Table 3: Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max and Min) as regressors, for each monkey", loc='center', wrap=True,fontweight='bold')
    ax.axis("off")
    fig.savefig(path_cwd + 'SupTables_2022.pdf')  # Finally save fig