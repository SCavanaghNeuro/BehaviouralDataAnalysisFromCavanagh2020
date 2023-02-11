import numpy as np
from scipy.optimize import minimize, fmin
from scipy.stats import chi2_contingency, norm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import \
    make_axes_locatable  # To make imshow 2D-plots to have color bars at the same height as the figure
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, IndexLocator
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy import stats
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import log_loss
import pandas as pd
from random import choices
from datetime import datetime
import h5py

matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)
from matplotlib.font_manager import FontProperties

font = FontProperties(family="Arial")
hfont = {'fontname': 'Arial'}
import matplotlib as mpl

mpl.rc('font', family='Arial')
mpl.rcParams['lines.linewidth'] = 1.5
plt.rcParams["font.family"] = "Arial"

# common variables across figures
max1 = 8.5 / 2.54  # involved in calculating figure size
max15 = 11.6 / 2.54
bar_width = 0.8
bar_opacity = 0.75
xbuf0 = 0.11  # x space between figure panels
ybuf0 = 0.08  # y space between figure panels
fontsize_fig_label = 10
fontsize_legend = 8
fontsize_tick = 7
ColorsHere = [(140. / 255, 81. / 255, 10. / 255), (128. / 255, 177. / 255, 211. / 255),
              (102. / 255, 102. / 255, 102. / 255)]
color_list_expt = [(33. / 255, 102. / 255, 172. / 255), (178. / 255, 24. / 255, 43. / 255)]

Reg_combined_color_list = [(0.4, 0.7607843137254902, 0.6470588235294118),
                           (0.9137254901960784, 0.6392156862745098, 0.788235294117647),
                           (0.4, 0.7607843137254902, 0.6470588235294118), 'grey', 'grey', 'grey', 'grey']

path_cwd = './'
FIGUREFileLocations = path_cwd + '/FigureFiles/'
DATAFileLocations = path_cwd + '/DataFiles/'

def MakeFig2(IndexOfTrToUse, EvidenceOptA, EvidenceOptB, Choices, TrialEr, MethodInput, Subject):
    dx_list, P_corr_Subj_list, ErrBar_P_corr_Subj_list, lik_model = \
        PsychometricFit(IndexOfTrToUse, EvidenceOptA, EvidenceOptB, Choices, TrialEr, MethodInput)

    x_values = 100. * dx_list[:-1]

    # calculate the weighting of evidence over time
    dm = np.concatenate(
        [np.ones([np.sum(IndexOfTrToUse), 1]), EvidenceOptA[IndexOfTrToUse, :] - EvidenceOptB[IndexOfTrToUse, :]],
        axis=1)  # create design matrix for logistic regression
    m = sm.Logit(Choices[IndexOfTrToUse], dm).fit(disp=0)  # Fit the model
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    ######## MAKE FIGURES

    ## Define subfigure domain.
    figsize = (max1, 1. * max1)

    # DEFINE POSITIONS OF FIGURE PANELS
    width1_11 = 0.32;
    width1_12 = width1_11
    width1_21 = width1_11;
    width1_22 = width1_21
    x1_11 = 0.135;
    x1_12 = x1_11 + width1_11 + 1.7 * xbuf0
    x1_21 = x1_11;
    x1_22 = x1_12
    height1_11 = 0.3;
    height1_12 = height1_11
    height1_21 = height1_11;
    height1_22 = height1_21
    y1_11 = 0.62;
    y1_12 = y1_11
    y1_21 = y1_11 - height1_21 - 2.35 * ybuf0;
    y1_22 = y1_21

    rect1_11_0 = [x1_11, y1_11, width1_11 * 0.05, height1_11]
    rect1_11 = [x1_11 + width1_11 * 0.2, y1_11, width1_11 * (1 - 0.2), height1_11]
    rect1_12_0 = [x1_12, y1_12, width1_12 * 0.05, height1_12]
    rect1_12 = [x1_12 + width1_12 * 0.2, y1_12, width1_12 * (1 - 0.2), height1_12]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]
    if Subject == 'Monkey H':
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
    ax_0 = fig_temp.add_axes(PsychBox_subfig)  # create a mini axis so the first datapoint is separate
    ax = fig_temp.add_axes(PsychBox)  # create the main axis for the figure
    remove_topright_spines(ax_0)  # remove top line from figure
    remove_topright_spines(ax)  # remove top line from figure
    ax.spines['left'].set_visible(False)
    remove_topright_spines(ax)  # remove top line from figure
    # Log-Spaced
    ax.errorbar(x_values[2:], P_corr_Subj_list[2:-1], ErrBar_P_corr_Subj_list[0, 2:-1], color='k',
                markerfacecolor='grey', ecolor='grey', fmt='.', zorder=4, clip_on=False, markeredgecolor='k',
                linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    tmp = ax_0.errorbar(x_values[1], P_corr_Subj_list[1], ErrBar_P_corr_Subj_list[0, 1], color='k',
                        markerfacecolor='grey', ecolor='grey', marker='.', zorder=4, clip_on=False, markeredgecolor='k',
                        linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6,
                        capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    for b in tmp[2]:
        b.set_clip_on(False)
    x_list_psychometric = np.arange(1, 50, 1)
    ax.plot(x_list_psychometric, PsychFitter(x_list_psychometric / 100, lik_model['x']), color='k', ls='-',
            clip_on=False, zorder=2)
    ax_0.scatter(0, PsychFitter(0, lik_model['x']), s=15., color='k', marker='_', clip_on=False, zorder=2,
                 linewidth=1.305)
    ax.set_xscale('log')
    ax.set_xlabel('Evidence for option', fontsize=fontsize_legend, x=0.4,
                  labelpad=1.)  # where to centre the XLabel, and distance from axis
    ax_0.set_ylabel('Accuracy', fontsize=fontsize_legend, labelpad=-3.)  # put the ylabel on the smaller axis
    ax_0.set_ylim([0.48, 1.])
    ax.set_ylim([0.48, 1.])
    ax_0.set_xlim([-1, 1])
    ax.set_xlim([1, 50])
    ax_0.set_xticks([0.])
    ax.xaxis.set_ticks([1, 10])
    ax_0.set_yticks([0.5, 1.])
    ax_0.yaxis.set_ticklabels([0.5, 1])

    minorLocator = MultipleLocator(0.1)
    ax_0.yaxis.set_minor_locator(minorLocator)
    ax.set_yticks([])  # remove ticks from the main axis
    ax_0.tick_params(direction='out', pad=1.5)
    ax_0.tick_params(which='minor', direction='out')
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')

    # set a break line between the two axes
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
    ax.errorbar(np.linspace(1, 8, 8), betas[1:], stdErs[1:], color='k', markerfacecolor='grey', ecolor='grey',
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

    if Subject == 'Monkey A':  # both subjects habe been processed, figure ready for saving
        fig_temp.savefig(FIGUREFileLocations + 'Figure2_2022.pdf')  # Finally save fig

def remove_topright_spines(ax):
    # hide the top and right spines
    [spin.set_visible(False) for spin in (ax.spines['top'], ax.spines['right'])]

    # hide the right and top tick marks
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    return None


def GetTrialDistributionData(EvidA, EvidB, TrialType, dx_distribution):
    x_distribution_list = np.arange(0, 1 + dx_distribution, dx_distribution)

    n_distribution_regularTr = np.zeros((len(x_distribution_list), len(x_distribution_list)))
    n_distribution_narrow_high = np.zeros((len(x_distribution_list), len(x_distribution_list)))
    n_distribution_broad_high = np.zeros((len(x_distribution_list), len(x_distribution_list)))
    n_distribution_NB_balanced = np.zeros((len(x_distribution_list), len(x_distribution_list)))

    dx_SD_distribution = 0.01  # Width of the bins used.
    x_SD_distribution_list = np.arange(0, 0.3 + dx_SD_distribution, dx_SD_distribution)  # List of bins used
    n_SD_distribution_regularTr = np.zeros((len(x_SD_distribution_list), len(x_SD_distribution_list)))

    MeanEvidA = np.mean(EvidA, axis=1)
    MeanEvidB = np.mean(EvidB, axis=1)
    MeanEvidDiff = np.mean(EvidA, axis=1) - np.mean(EvidB, axis=1)
    SdEvidA = np.std(EvidA, axis=1, ddof=1)
    SdEvidB = np.std(EvidB, axis=1, ddof=1)
    SdEvidDiff = np.std(EvidA, axis=1, ddof=1) - np.std(EvidB, axis=1, ddof=1)

    for i in range(0, len(SdEvidDiff)):
        if TrialType[i] == 1:
            if SdEvidA[i] >= SdEvidB[i]:
                n_distribution_regularTr[np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1,
                                         np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1] = \
                    n_distribution_regularTr[np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1,
                                             np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1] + 1

                n_SD_distribution_regularTr[np.ceil(SdEvidB[i] / dx_SD_distribution).astype(int) - 1, np.ceil(
                    SdEvidA[i] / dx_SD_distribution).astype(int) - 1] = \
                    n_SD_distribution_regularTr[np.ceil(SdEvidB[i] / dx_SD_distribution).astype(int) - 1, np.ceil(
                        SdEvidA[i] / dx_SD_distribution).astype(int) - 1] + 1

            else:
                n_distribution_regularTr[np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1,
                                         np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1] = \
                    n_distribution_regularTr[np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1,
                                             np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1] + 1;

                n_SD_distribution_regularTr[np.ceil(SdEvidA[i] / dx_SD_distribution).astype(int) - 1, np.ceil(
                    SdEvidB[i] / dx_SD_distribution).astype(int) - 1] = \
                    n_SD_distribution_regularTr[np.ceil(SdEvidA[i] / dx_SD_distribution).astype(int) - 1, np.ceil(
                        SdEvidB[i] / dx_SD_distribution).astype(int) - 1] + 1

        elif TrialType[i] == 18:
            n_distribution_narrow_high[
                np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1, np.ceil(MeanEvidA[i] / dx_distribution).astype(
                    int) - 1] = \
                n_distribution_narrow_high[np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1,
                                           np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1] + 1
        elif TrialType[i] == 21:
            n_distribution_narrow_high[
                np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1, np.ceil(MeanEvidB[i] / dx_distribution).astype(
                    int) - 1] = \
                n_distribution_narrow_high[np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1,
                                           np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1] + 1
        elif TrialType[i] == 19:
            n_distribution_broad_high[
                np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1, np.ceil(MeanEvidA[i] / dx_distribution).astype(
                    int) - 1] = \
                n_distribution_broad_high[np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1,
                                          np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1] + 1
        elif TrialType[i] == 22:
            n_distribution_broad_high[
                np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1, np.ceil(MeanEvidB[i] / dx_distribution).astype(
                    int) - 1] = \
                n_distribution_broad_high[np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1,
                                          np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1] + 1
        elif TrialType[i] == 20:
            n_distribution_NB_balanced[
                np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1, np.ceil(MeanEvidA[i] / dx_distribution).astype(
                    int) - 1] = \
                n_distribution_NB_balanced[np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1,
                                           np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1] + 1
        elif TrialType[i] == 23:
            n_distribution_NB_balanced[
                np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1, np.ceil(MeanEvidB[i] / dx_distribution).astype(
                    int) - 1] = \
                n_distribution_NB_balanced[np.ceil(MeanEvidA[i] / dx_distribution).astype(int) - 1,
                                           np.ceil(MeanEvidB[i] / dx_distribution).astype(int) - 1] + 1

    return n_distribution_regularTr, n_distribution_narrow_high, n_distribution_broad_high, n_distribution_NB_balanced, n_SD_distribution_regularTr


def Analyse_NarBroad(ChosenTarget, TrialType, EvidA, EvidB):
    # Define the trial types
    BroadLow_Versus_NarrowHigh = TrialType == 18
    BroadHigh_Versus_NarrowLow = TrialType == 19
    BroadBalanced_Versus_NarrowBalanced = TrialType == 20
    NarrowHigh_Versus_BroadLow = TrialType == 21
    NarrowLow_Versus_BroadHigh = TrialType == 22
    NarrowBalanced_Versus_BroadBalanced = TrialType == 23

    NoTr = [(np.sum(BroadLow_Versus_NarrowHigh) + np.sum(NarrowHigh_Versus_BroadLow)), \
            (np.sum(NarrowLow_Versus_BroadHigh) + np.sum(BroadHigh_Versus_NarrowLow)), \
            (np.sum(NarrowBalanced_Versus_BroadBalanced) + np.sum(BroadBalanced_Versus_NarrowBalanced))]

    # Calculate the accuracy and standard error on each of the two trial types
    Accuracy = np.zeros((2, 1))
    StdErs = np.zeros((3, 1))

    Accuracy[0] = (np.sum(ChosenTarget[BroadLow_Versus_NarrowHigh] == 0) + np.sum(
        ChosenTarget[NarrowHigh_Versus_BroadLow] == 1)) / NoTr[0]
    Accuracy[1] = (np.sum(ChosenTarget[NarrowLow_Versus_BroadHigh] == 0) + np.sum(
        ChosenTarget[BroadHigh_Versus_NarrowLow] == 1)) / NoTr[1]
    StdErs[0:2] = np.sqrt((Accuracy * (1 - Accuracy)).T / NoTr[:-1]).T

    # Calculate broad preference and standard error on balanced trials
    BroadPref = (np.sum(ChosenTarget[NarrowBalanced_Versus_BroadBalanced] == 0) + np.sum(
        ChosenTarget[BroadBalanced_Versus_NarrowBalanced] == 1)) / NoTr[2]
    StdErs[2] = np.sqrt((BroadPref * (1 - BroadPref)).T / NoTr[-1:]).T

    # Compare accuracy on NarrowHigh and BroadHigh trials using Chi-2 FigureFiles
    ChiInputRow1 = [(np.sum(ChosenTarget[BroadLow_Versus_NarrowHigh] == 0) + np.sum(
        ChosenTarget[NarrowHigh_Versus_BroadLow] == 1)), (
                            np.sum(ChosenTarget[NarrowLow_Versus_BroadHigh] == 0) + np.sum(
                        ChosenTarget[BroadHigh_Versus_NarrowLow] == 1))]

    ChiInput = [ChiInputRow1, (np.array(NoTr[0:2]) - ChiInputRow1).tolist()]
    stat, p, dof, expected = chi2_contingency(ChiInput, correction=False)

    # Compare broad preference on ambigous trials with chance using binomial FigureFiles
    BinomP = stats.binom_test((np.sum(ChosenTarget[NarrowBalanced_Versus_BroadBalanced] == 0) + np.sum(
        ChosenTarget[BroadBalanced_Versus_NarrowBalanced] == 1)), NoTr[2], p=0.5, alternative='two-sided')

    # visualise information about the trial difficulty
    dx_distribution = 0.0005
    distRegTr, distNarHighTr, distBroadHighTr, distBalancedTr, distSdRegTr \
        = GetTrialDistributionData(EvidA, EvidB, TrialType, dx_distribution)

    density_distribution_narrow_high_all = (distNarHighTr) / np.sum(distNarHighTr)
    density_distribution_broad_high_all = (distBroadHighTr) / np.sum(distBroadHighTr)
    density_distribution_NB_balanced_all = (distBalancedTr) / np.sum(distBalancedTr)
    density_distribution_net_narrow_high_all = np.zeros(len(density_distribution_narrow_high_all))
    density_distribution_net_broad_high_all = np.zeros(len(density_distribution_broad_high_all))
    density_distribution_net_NB_balanced_all = np.zeros(len(density_distribution_NB_balanced_all))

    # get a distribution for NET evidence (i.e. 1D from the 2D matrix)
    for i in range(len(density_distribution_net_narrow_high_all)):
        density_distribution_net_narrow_high_all[i] = np.sum(density_distribution_narrow_high_all.diagonal(
            i - int((len(density_distribution_narrow_high_all) - 1.) / 2.)))
        density_distribution_net_broad_high_all[i] = np.sum(
            density_distribution_broad_high_all.diagonal(i - int((len(density_distribution_broad_high_all) - 1.) / 2.)))
        density_distribution_net_NB_balanced_all[i] = np.sum(density_distribution_NB_balanced_all.diagonal(
            i - int((len(density_distribution_NB_balanced_all) - 1.) / 2.)))

    # smooth the net evidence distributions
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
    width1_11 = 0.8
    width1_22 = 0.185
    width1_21 = width1_22 * (1 + bar_width) / bar_width
    width1_31 = width1_21
    width1_32 = width1_22
    x1_11 = 0.14
    x1_21 = x1_11 - 0.18 * xbuf0
    x1_22 = x1_21 + width1_21 + 1.8 * xbuf0
    x1_31 = x1_21
    x1_32 = x1_22
    height1_11 = 0.22
    height1_21 = height1_11
    height1_22 = height1_21
    height1_31 = height1_21
    height1_32 = height1_22
    y1_11 = 0.725
    y1_21 = y1_11 - height1_21 - 1.5 * ybuf0
    y1_22 = y1_21
    y1_31 = y1_21 - height1_31 - 1.25 * ybuf0
    y1_32 = y1_31

    rect1_11 = [x1_11, y1_11, width1_11, height1_11]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]
    rect1_31 = [x1_31, y1_31, width1_31, height1_31]
    rect1_32 = [x1_32, y1_32, width1_32, height1_32]

    ##### Plot the figure letters and panel titles
    fig_temp = plt.figure(num=3, figsize=figsize)
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

    # Plot the first panel
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
    for color, text, item in zip([ColorsHere[0], ColorsHere[2], ColorsHere[1]], legend.get_texts(),
                                 legend.legendHandles):
        text.set_color(color)
        item.set_visible(False)
    ax.text(-16.5, 0.0105, r'$\times \mathregular{10^{-2}}$', fontsize=fontsize_tick - 1.)

    # Store the key variables to make a for loop across panels with same design
    PanelLocation = np.concatenate((rect1_21, rect1_22, rect1_31, rect1_32)).reshape(2, *(2, 4))

    DataFromHumanStudy = np.array([0.671, 0.781, 0.623, ])
    ErFromHumanStudy = np.array([0.065 / 2., 0.08979 / 2., 0.09598 / 2.])

    BarHeights = [np.append(Accuracy, BroadPref), DataFromHumanStudy]
    BarErs = [StdErs, ErFromHumanStudy]

    for ii in [0, 1]:
        ##Plot the bar charts - first subplot
        ax = fig_temp.add_axes(PanelLocation[ii, 0, :])
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
        ax = fig_temp.add_axes(PanelLocation[ii, 1, :])
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

    fig_temp.savefig(FIGUREFileLocations + 'Figure3_2022.pdf')  # Finally save fig
    print('     Completed Figure 3')


def sliding_win_on_lin_data(data_mat, window_width, axis=0):
    smaller_half = np.floor(window_width / 2)
    bigger_half = np.ceil(window_width / 2)
    data_mat_result = np.zeros(len(data_mat))
    for k_lin in range(len(data_mat)):
        lower_bound = math.floor(np.maximum(k_lin - smaller_half, 0))
        upper_bound = math.floor(np.minimum(k_lin + bigger_half, len(data_mat)))
        data_mat_result[k_lin] = np.mean(data_mat[lower_bound:upper_bound])
    return data_mat_result


def PsychometricFit(IndexOfTrToUse, EvidenceOptA, EvidenceOptB, Choices, TrialEr, MethodInput):
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
        idx_pos_log_Collapsed = np.zeros((sum(IndexOfTrToUse)))
        EqualValueTr = (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))

        idx_pos_log_Collapsed[EqualValueTr==0] = dx_pos_log_list.shape[0] - np.round((np.log(np.abs(
            np.mean(EvidenceOptA[IndexOfTrToUse, :][EqualValueTr==0], axis=1) - np.mean(EvidenceOptB[IndexOfTrToUse, :][EqualValueTr==0],
                                                                       axis=1))) - np.log(dx_list[-1])) \
                                                                    / (np.log(dx_pos_log_list[0]) - np.log(
            dx_pos_log_list[-1])) * (np.shape(dx_pos_log_list)[0] - 1))
        # Log-Spaced. Ignoring signs and only map to positive log-space (from 1 to length(dx_pos_log_list)).

        idx_Collapsed = idx_pos_log_Collapsed + 2
        idx_Collapsed[
            idx_pos_log_Collapsed < 1] = 2  # 1 in i_dx_pos_log_Collapsed => absolute diff in evidence is less than the minimum for dx_pos_log_list(1)=0.02 (~0.017).
        idx_Collapsed[EqualValueTr] = 1

    elif MethodInput == 'NarrowBroad':
        is_A_broad = (2 * np.heaviside(
            EvidenceOptA[IndexOfTrToUse, :].std(axis=1) - EvidenceOptB[IndexOfTrToUse, :].std(axis=1), 0.5)) - 1
        is_A_broad[is_A_broad == 0] = 1

        EqualValueTr = (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))
        i_dx_pos_log_Collapsed_NB = np.zeros((sum(IndexOfTrToUse)))
        i_dx_pos_log_Collapsed_NB[EqualValueTr==0] = len(dx_pos_log_list) - np.round( \
            (np.log(np.abs(EvidenceOptA[IndexOfTrToUse, :][EqualValueTr==0] .mean(axis=1) - \
                           EvidenceOptB[IndexOfTrToUse, :][EqualValueTr==0] .mean(axis=1))) \
             - np.log(dx_pos_log_list[-1])) \
            / (np.log(dx_pos_log_list[0]) - np.log(dx_pos_log_list[-1])) * (len(dx_pos_log_list) - 1))

        idx_Collapsed = i_dx_pos_log_Collapsed_NB + 1
        idx_Collapsed[i_dx_pos_log_Collapsed_NB < 1] = 1
        idx_Collapsed[EqualValueTr] = 0
        idx_Collapsed = np.sign([np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) - np.mean(
            EvidenceOptB[IndexOfTrToUse, :], axis=1)]) * idx_Collapsed
        idx_Collapsed = idx_Collapsed * is_A_broad + len(dx_pos_log_list) + 2;
        idx_Collapsed = idx_Collapsed.reshape(-1)

       # Setting up the Y-variable
    if MethodInput == 'CorrectIncorrect':  # Y variable is whether the correct option was chosen
        ChoseCorrect = (BestOption.reshape(-1) == 1 * Choices) + (BestOption.reshape(-1) == 2 * (Choices == 0))
        ChoseCorrect[
            (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :], axis=1))] = \
            TrialEr[
                (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) == np.mean(EvidenceOptB[IndexOfTrToUse, :],
                                                                             axis=1))] == 0
        YVar = ChoseCorrect
    elif MethodInput == 'NarrowBroad':
        Broad_Option_Chosen_Collapsed = 0.5 * (is_A_broad + 1) * (Choices) \
                                        + -0.5 * (is_A_broad - 1) * (1 - (Choices))
        YVar = Broad_Option_Chosen_Collapsed

    # For loop across bins
    for i in range(1, np.shape(dx_list)[0]):
        n_dx_list[i - 1] = np.sum(idx_Collapsed == i)
        if n_dx_list[i - 1]>0:
            P_corr_Subj_list[i - 1] = np.sum(YVar[idx_Collapsed == i]) / np.sum(idx_Collapsed == i)
        else:
            P_corr_Subj_list[i - 1] = 0

            # Calculate errorbars
    ErrBar_P_corr_Subj_list = np.zeros((1,len(n_dx_list)))
    ErrBar_P_corr_Subj_list[0,n_dx_list>0] = np.sqrt(P_corr_Subj_list[n_dx_list>0] * [1 - P_corr_Subj_list[n_dx_list>0]] / n_dx_list[n_dx_list>0])
    ErrBar_P_corr_Subj_list[0, n_dx_list == 0] = float("nan")

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
        evidence_list = (np.mean(EvidenceOptA[IndexOfTrToUse, :], axis=1) - np.mean(
                            EvidenceOptB[IndexOfTrToUse, :], axis=1)) * is_A_broad

        def MleFitSC_NB(params):
            a = params[0]
            b = params[1]
            c = params[2]

            preds = 0.5 + 0.5 * np.sign(evidence_list + c) * (1. - np.exp(
                -np.power((np.abs(evidence_list + c) / a), b)))

            return -np.sum(np.log((YVar * preds) + ((1 - YVar) * (1 - preds))))

        initParams = [0.05, 1, 0]
        lik_model = minimize(MleFitSC_NB, initParams, method='Nelder-Mead')

    return dx_list, P_corr_Subj_list, ErrBar_P_corr_Subj_list, lik_model

def PsychFitter(evidence_list, params):
    a = params[0]
    b = params[1]
    return 0.5 + 0.5 * (1 - np.exp(-np.power((evidence_list / a), b)))

def PsychFitterNB(evidence_list, params):
    a = params[0]
    b = params[1]
    c = params[2]
    return 0.5 + 0.5 * np.sign(evidence_list + c) * (1. - np.exp(
        -np.power((np.abs(evidence_list + c) / a), b)))

def MakeFig4(Choices, EvidenceOptA, EvidenceOptB, TrialEr, TrialType, MethodInput):
    # Fit the psychometric function split by higher SD and lower SD option correct

    dx_list, P_corr_Subj_list, ErrBar_P_corr_Subj_list, lik_model = \
        PsychometricFit(np.ones(len(Choices)) > 0, EvidenceOptA, EvidenceOptB, Choices, TrialEr, MethodInput)

    # Fit a logistic regression model with Mean Evidence difference and Std Evidence difference as regressors

    Regressor1 = ((np.mean(EvidenceOptA, axis=1) - np.mean(EvidenceOptB, axis=1)).reshape(len(EvidenceOptB), 1))
    Regressor2 = ((np.std(EvidenceOptA, axis=1, ddof=1) - np.std(EvidenceOptB, axis=1, ddof=1)).reshape(len(EvidenceOptB), 1))
    dm = np.concatenate((np.ones((len(EvidenceOptA), 1)), Regressor1, Regressor2), axis=1)

    m = sm.Logit(Choices == 1, dm).fit(disp=0)
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    ## Extract number distribution of stimuli, for Standard/Regression trials.
    dx_Reg_density = 0.1
    n_x_Reg_smooth = 20

    dx_distribution = 0.02  # Width of the bins used.
    distRegTr, distNarHighTr, distBroadHighTr, distBalancedTr, distSdRegTr \
        = GetTrialDistributionData(EvidenceOptA, EvidenceOptB, TrialType, dx_distribution)

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

    # Make the figures
    x_list_psychometric = np.arange(0.01, 0.5, 0.01)
    x0_psychometric = 0
    ## Define subfigure domain.
    figsize = (max1, 1.2 * max1)

    width1_11 = 0.3
    width1_12 = 0.25
    width1_21 = 0.3
    width1_22 = 0.25
    x1_11 = 0.15
    x1_12 = x1_11 + width1_12 + 2.6 * xbuf0
    x1_21 = x1_11
    x1_22 = x1_12 - 0.021
    height1_11 = 0.3
    height1_12 = 0.24
    height1_21 = height1_11
    height1_22 = 0.28
    y1_11 = 0.59
    y1_12 = y1_11 + 0.038
    y1_21 = y1_11 - height1_21 - 2.4 * ybuf0
    y1_22 = y1_21 + 0.013

    rect1_11 = [x1_11, y1_11, width1_11, height1_11]
    rect1_12_0 = [x1_12, y1_12, width1_12 * 0.05, height1_12]
    rect1_12 = [x1_12 + width1_12 * 0.2, y1_12, width1_12 * (1 - 0.2), height1_12]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]

    ##### Plot the figure labels
    fig_temp = plt.figure(num=4, figsize=figsize)
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
    tmp = ax_0.errorbar(-d_evidence_avg_list[9], 1. - P_corr_avg_list[9], ErrBar_P_corr_avg_list[9],
                        color=ColorsHere[0],
                        ecolor=ColorsHere[0], marker='.', zorder=3, clip_on=False, markeredgecolor='k', linewidth=0.3,
                        elinewidth=0.6, markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    for b in tmp[2]:
        b.set_clip_on(False)
    ax.plot(100. * x_list_psychometric, PsychFitterNB(x_list_psychometric, lik_model.x),
            color=ColorsHere[1], ls='-', clip_on=False)  # , linestyle=linestyle_list[i_var_a])
    ax.plot(100. * x_list_psychometric,
            1. - PsychFitterNB(-x_list_psychometric, lik_model.x), color=ColorsHere[0],
            ls='-', clip_on=False)  # , linestyle=linestyle_list[i_var_a])
    ax_0.scatter(100. * x0_psychometric, PsychFitterNB(x0_psychometric, lik_model.x),
                 s=15., color=ColorsHere[1], marker='_', clip_on=False,
                 linewidth=1.305)  # , linestyle=linestyle_list[i_var_a])
    ax_0.scatter(100. * x0_psychometric,
                 1. - PsychFitterNB(-x0_psychometric, lik_model.x), s=15.,
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

    fig_temp.savefig(FIGUREFileLocations + 'Figure4_2022.pdf')  # Finally save fig
    print('     Completed Figure 4')


def AdvancedRegrAnalysis(Choices, EvidA, EvidB, BootStrapNo, n_kcv_runs):
    # Make an initial design matrix
    dm = np.stack((np.ones(len(EvidA)), \
                   EvidA[:, 0], EvidA[:, -1], np.mean(EvidA, axis=1), np.max(EvidA, axis=1), np.min(EvidA, axis=1), \
                   EvidB[:, 0], EvidB[:, -1], np.mean(EvidB, axis=1), np.max(EvidB, axis=1), np.min(EvidB, axis=1))).T
    m = sm.Logit(Choices, dm).fit(disp=0)
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    # specify the full design matrix, then specify several nested models
    dm_full = np.stack((np.ones(len(EvidA)), \
                        EvidA[:, 0], EvidA[:, -1], np.mean(EvidA, axis=1), np.std(EvidA, axis=1, ddof=1),
                        np.max(EvidA, axis=1), np.min(EvidA, axis=1), \
                        EvidB[:, 0], EvidB[:, -1], np.mean(EvidB, axis=1), np.std(EvidB, axis=1, ddof=1),
                        np.max(EvidB, axis=1), np.min(EvidB, axis=1))).T

    Listofallmodels = [dm_full[:, (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12)],  # No SD
                       dm_full,  # Full model
                       dm_full[:, (0, 1, 2, 3, 4, 7, 8, 9, 10)],  # No max/min
                       dm_full[:, (0, 3, 4, 5, 6, 9, 10, 11, 12)],  # no first/last
                       dm_full[:, (0, 3, 4, 9, 10)],  # mean and SD
                       dm_full[:, (0, 3, 5, 6, 9, 11, 12)],  # mean max min
                       dm_full[:, (0, 1, 2, 3, 7, 8, 9)],  # First last mean
                       dm_full[:, (0, 3, 9)],  # mean only
                       dm_full[:, (0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12)]]  # No mean
    # Set up the cross-validation
    rkf = RepeatedKFold(n_splits=10, n_repeats=n_kcv_runs)
    TotalCost = np.zeros(len(Listofallmodels))
    RunsComplete = 0
    for train, test in rkf.split(Choices):
        for ii in range(0, len(Listofallmodels)):
            dm = Listofallmodels[ii]
            m = sm.Logit(Choices[train], dm[train, :]).fit(disp=0)
            TotalCost[ii] = TotalCost[ii] + (
                        (-log_loss(Choices[test], m.predict(dm[test, :])) * len(test)) / rkf.n_repeats)
        RunsComplete = RunsComplete + 1
        if np.remainder((RunsComplete / n_kcv_runs * 10), 10) == 0:
            print('         Completed ' + str(RunsComplete / n_kcv_runs * 10) + ' % of standard day cross-validation analysis')

    # format the output as tables
    OutputValues = [None] * 3
    # table 1 = Difference in log-likelihood of Full regression model (mean, SD, max, min, first, last of evidence values; equation 6 in Methods) vs reduced model
    OutputValues[0] = np.array([TotalCost[1] - TotalCost[8], TotalCost[1] - TotalCost[3], TotalCost[1] - TotalCost[0],
                                TotalCost[1] - TotalCost[2]])
    # table 2 = Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max & Min) as regressors, for each monkey and the circuit model
    OutputValues[1] = np.array([TotalCost[4] - TotalCost[5], TotalCost[2] - TotalCost[0]])
    # table 3 = Increase in log-likelihood of various regression models (regressors in column labels) due to inclusion of evidence standard deviation as a regressor
    OutputValues[2] = np.array([TotalCost[4] - TotalCost[7], TotalCost[2] - TotalCost[6], TotalCost[3] - TotalCost[5],
                                TotalCost[1] - TotalCost[0]])

    return OutputValues


def MakeSupTables(OutputValues, Subjects):
    fig = plt.figure(num=5, figsize=(9, 6))
    DefFontSize = 8
    rectPos = [[0.1, 0.65, 0.8, 0.2],
               [0.1, 0.35, 0.8, 0.2],
               [0.1, 0.05, 0.8, 0.2]]

    df = pd.DataFrame(np.concatenate((OutputValues[0][0], OutputValues[1][0])).reshape(2, 4),
                      columns=['Mean', 'First/Last', 'SD', 'Max/min'], index=Subjects).round(1)


    ax = fig.add_axes(rectPos[0])
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     loc="center left")
    table.auto_set_font_size(False)
    table.set_fontsize(DefFontSize)
    ax.set_title(
        "Supplementary Table 1: Difference in log-likelihood of Full regression model (mean, SD, max, min, first, last of evidence values; Equation 6 in Materials and methods) vs reduced model, for each monkey",
        loc='center', wrap=True, fontweight='bold')
    ax.axis("off")

    df = pd.DataFrame(np.concatenate((OutputValues[0][1], OutputValues[1][1])).reshape(2, 2),
                      columns=['Mean', 'Mean & First & Last'], index=Subjects).round(1)
    ax = fig.add_axes(rectPos[1])
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     loc="center left",
                     colWidths=[0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(DefFontSize)
    ax.set_title(
        "Supplementary Table 2: Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max and Min) as regressors, for each monkey",
        loc='center', wrap=True, fontweight='bold')
    ax.axis("off")

    df = pd.DataFrame(np.concatenate((OutputValues[0][2], OutputValues[1][2])).reshape(2, 4),
                      columns=['Mean', 'Mean, First & Last', 'Mean, Max & Min', 'Mean, Max, Min, First & Last'],
                      index=Subjects).round(1)
    ax = fig.add_axes(rectPos[2])
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     loc="center left")
    table.auto_set_font_size(False)
    table.set_fontsize(DefFontSize)
    ax.set_title(
        "Supplementary Table 3: Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max and Min) as regressors, for each monkey",
        loc='center', wrap=True, fontweight='bold')
    ax.axis("off")
    fig.savefig(FIGUREFileLocations + 'SupTables_2022.pdf')  # Finally save fig
    print('     Completed supplementary tables')

def AnalyseWithinSessionDrugDayData(WithinSessionStruct):
    # Extract key variables
    resp_trials = WithinSessionStruct['resp_trials'][:,
                  0]  # Did the subject complete the trial by making a behavioural choice?
    TrialError = WithinSessionStruct['TrialError'][:,
                 0]  # MonkeyLogic record of trial outcome (0 = correct; 6 = incorrect; others = imply incomplete)
    DrugGivenTime = WithinSessionStruct['DrugGivenTime'][0, :]  # Time in HH; MM that the subject was injected
    TrialStarTimeHours = WithinSessionStruct['TrialStarTimeHours'][:, 0]  # Time each trial began - hours
    TrialStarTimeMins = WithinSessionStruct['TrialStarTimeMins'][:, 0]  # Time each trial began - minutes
    TrialStarTimeSecs = WithinSessionStruct['TrialStarTimeSecs'][:, 0]  # Time each trial began - seconds
    TrialType = WithinSessionStruct['CompletedTrialType'][0, :].astype(
        int)  # Reference number of the trial type for further anaylses
    ChosenTarget = WithinSessionStruct['ChosenTarget'][0, :]  # Chosen target (1=left; 2=right) on Completed trials
    EvidenceUnitsA = WithinSessionStruct['EvidenceUnitsA'][:,:]  # Evidence values for the Left Option on Completed trials
    EvidenceUnitsB = WithinSessionStruct['EvidenceUnitsB'][:,:]  # Evidence values for the Right Option on Completed trials

    # Work out the times of injection, Drug ON (5 mins post injection), and Drug OFF (30 mins post injection).
    SecondsIntoDayDrugGiven = DrugGivenTime[0] * 60 * 60 + DrugGivenTime[1] * 60
    SecondsIntoDayDrugEffective = SecondsIntoDayDrugGiven + [5 * 60]
    SecondsIntoDayDrugOver = SecondsIntoDayDrugGiven + [30 * 60]
    SecondsIntoDayTrialsStart = (TrialStarTimeHours.astype('int') * 60 * 60) + (
                TrialStarTimeMins.astype('int') * 60) + TrialStarTimeSecs

    DrugTrials = np.zeros((len(SecondsIntoDayTrialsStart), 1))
    PreDrugTrials = np.zeros((len(SecondsIntoDayTrialsStart), 1))
    PostDrugTrials = np.zeros((len(SecondsIntoDayTrialsStart), 1))

    DrugTrials[(SecondsIntoDayTrialsStart > SecondsIntoDayDrugEffective) * (
                SecondsIntoDayTrialsStart < SecondsIntoDayDrugOver)] = 1
    PreDrugTrials[SecondsIntoDayTrialsStart < SecondsIntoDayDrugGiven] = 1
    PostDrugTrials[SecondsIntoDayTrialsStart > SecondsIntoDayDrugOver] = 1

    ## Select the trials to include in the analyses (Responded AND trials of a certain trial type).
    DrugTrialsResp = DrugTrials[resp_trials == 1]  # On drug trials, where the subject responded
    PreDrugTrialsResp = PreDrugTrials[resp_trials == 1]  # Pre drug trials, where the subject responded
    PostDrugTrialsResp = PostDrugTrials[resp_trials == 1]
    TrialErrorResp = TrialError[resp_trials == 1]  # Trial error, just on responded trials

    TrialsToIncludeInAnalysesForDrugDays = (TrialType == 1) + (TrialType == 16) + (TrialType == 17) + (
                TrialType == 20) + (TrialType == 23)
    DrugTrialsResp_IncTrials = DrugTrialsResp[TrialsToIncludeInAnalysesForDrugDays]
    PreDrugTrialsResp_IncTrials = PreDrugTrialsResp[TrialsToIncludeInAnalysesForDrugDays]
    PostDrugTrialsResp_Reg_Trials = PostDrugTrialsResp[TrialsToIncludeInAnalysesForDrugDays]

    # Redefine these variables to just include elements from included trials
    ChosenTarget_IncTrials = ChosenTarget[TrialsToIncludeInAnalysesForDrugDays]
    EvidenceUnitsA_IncTrials = EvidenceUnitsA[TrialsToIncludeInAnalysesForDrugDays, :]
    EvidenceUnitsB_IncTrials = EvidenceUnitsB[TrialsToIncludeInAnalysesForDrugDays, :]
    TrialErrorResp_IncTrials = TrialErrorResp[TrialsToIncludeInAnalysesForDrugDays]
    DrugTrialsResp_TrialError = TrialErrorResp_IncTrials[(DrugTrialsResp_IncTrials == 1).reshape(-1)]

    ## 9 regerssors model to analyse pro-variance bias (3 constant terms, 3 regressors with evidence mean, 3 regressors with STD information)

    DM_Here = np.hstack((PreDrugTrialsResp_IncTrials, DrugTrialsResp_IncTrials, PostDrugTrialsResp_Reg_Trials, \
                         (np.mean(EvidenceUnitsA_IncTrials, axis=1) - np.mean(EvidenceUnitsB_IncTrials,
                                                                              axis=1)).reshape(
                             len(EvidenceUnitsA_IncTrials), 1), \
                         (np.std(EvidenceUnitsA_IncTrials, axis=1, ddof=1) - np.std(EvidenceUnitsB_IncTrials, axis=1,
                                                                                    ddof=1)).reshape(
                             len(EvidenceUnitsA_IncTrials),
                             1)))  # Up to this point, columns 1 to 3 are constant terms; %4 is the Mean Evidence difference between Left and Right options; %5 is the STD difference between Left and Right Options
    DM_Here = np.hstack((DM_Here, np.vstack(
        ((DM_Here[:, 0] * DM_Here[:, 3]), (DM_Here[:, 1] * DM_Here[:, 3]), (DM_Here[:, 2] * DM_Here[:, 3]))).T))
    DM_Here = np.hstack((DM_Here, np.vstack(
        ((DM_Here[:, 0] * DM_Here[:, 4]), (DM_Here[:, 1] * DM_Here[:, 4]), (DM_Here[:, 2] * DM_Here[:, 4]))).T))

    ProVar9RegModel = np.hstack(
        (DM_Here[:, (0, 1, 2, 5, 6, 7, 8, 9, 10)], (ChosenTarget_IncTrials == 1).reshape(len(DM_Here), 1)))
    SessionWiseDMs = {"ProVar9RegModel": ProVar9RegModel}

    ## Store a regression model to look at the temporal weights
    SessionWiseDMs["PreDrugTemporalWeightsDM"] = \
        np.hstack((EvidenceUnitsA_IncTrials[(PreDrugTrialsResp_IncTrials == 1).reshape(-1), :],
                   # Columns 1-6 are the evidence on the left hand side, during predrug trials
                   EvidenceUnitsB_IncTrials[(PreDrugTrialsResp_IncTrials == 1).reshape(-1), :],
                   # Columns 7-12 are the evidence on the right hand side, during predrug trials
                   ChosenTarget_IncTrials[(PreDrugTrialsResp_IncTrials == 1).reshape(-1)].reshape(
                       np.sum(PreDrugTrialsResp_IncTrials == 1), 1)))  # Column 13 is the choice responses.

    SessionWiseDMs["OnDrugTemporalWeightsDM"] = \
        np.hstack((EvidenceUnitsA_IncTrials[(DrugTrialsResp_IncTrials == 1).reshape(-1), :],
                   # Columns 1-6 are the evidence on the left hand side, during drug trials
                   EvidenceUnitsB_IncTrials[(DrugTrialsResp_IncTrials == 1).reshape(-1), :],
                   # Columns 7-12 are the evidence on the right hand side, during drug trials
                   ChosenTarget_IncTrials[(DrugTrialsResp_IncTrials == 1).reshape(-1)].reshape(
                       np.sum(DrugTrialsResp_IncTrials == 1), 1)))  # Column 13 is the choice responses.
    ## Store information about the trial errors on completed trials
    SessionWiseDMs[
        "CompTr_TrialError"] = DrugTrialsResp_TrialError  # Store the trial error information on completed trials

    ## Store trial information in sliding bins relative to the time of injection
    ProVar_FullDM = np.vstack((np.mean(EvidenceUnitsA_IncTrials, axis=1) - np.mean(EvidenceUnitsB_IncTrials, axis=1),
                               np.std(EvidenceUnitsA_IncTrials, axis=1, ddof=1) - np.std(EvidenceUnitsB_IncTrials,
                                                                                         axis=1,
                                                                                         ddof=1))).T  # Pro-variance regression model, for all completed trials
    ChosenTargetMaker = (ChosenTarget_IncTrials == 1)  # Chosen target, for all completed trials

    BinWidthMins = 6
    StepSize = 1  # In minutes
    StartPoint = -20  # Minutes relative to injection to start the analysis
    EndPoint = 60  # Minutes relative to injection to end the analysis
    AllBins = np.arange(StartPoint, EndPoint + StepSize, StepSize)
    BinCounter = 0  # For use in for loop below
    BinnedPerf = np.zeros((1, len(AllBins)))
    OutputDMHere = [None] * len(AllBins)

    for Bn in AllBins:  # Loop across bins
        BinStartTime = SecondsIntoDayDrugGiven + (60 * Bn) - (60 * BinWidthMins / 2)  # In seconds from midnight
        BinEndTime = SecondsIntoDayDrugGiven + (60 * Bn) + (60 * BinWidthMins / 2)

        TrInThisBin = (SecondsIntoDayTrialsStart > BinStartTime) * (
                    SecondsIntoDayTrialsStart < BinEndTime)  # Find trials started in this bin
        TrInThisBinRelevant = TrInThisBin[resp_trials == 1][
            TrialsToIncludeInAnalysesForDrugDays == 1]  # Find completed included trials

        # Performance accuracy in this bin
        if np.sum(TrInThisBinRelevant)>0:
            BinnedPerf[0, BinCounter] = np.sum(TrialErrorResp_IncTrials[TrInThisBinRelevant == 1] == 0) / (
                        np.sum(TrialErrorResp_IncTrials[TrInThisBinRelevant == 1] == 0) + np.sum(
                    TrialErrorResp_IncTrials[TrInThisBinRelevant == 1] == 6))
        else:
            BinnedPerf[0, BinCounter] = float('nan')

        # Output the pro-variance bias regression design matrix here
        OutputDMHere[BinCounter] = np.hstack((ProVar_FullDM[TrInThisBinRelevant, :],
                                              ChosenTargetMaker[TrInThisBinRelevant].reshape(
                                                  np.sum(TrInThisBinRelevant), 1)))

        BinCounter = BinCounter + 1
    ## Reference all trials - according to whether they are pre, on, or post drug
    NewVarForTrRef = np.zeros((len(DrugTrials), 1))
    NewVarForTrRef[PreDrugTrials == 1] = 1
    NewVarForTrRef[DrugTrials == 1] = 2
    NewVarForTrRef[PostDrugTrials == 1] = 3
    SessionWiseDMs["NewVarForTrRef"] = NewVarForTrRef
    SessionWiseDMs["resp_trials"] = resp_trials
    return SessionWiseDMs, BinnedPerf, OutputDMHere

def FindSessionsToInclude(SessionWiseDMs, OverallDataStructure):
    FileNameForSessions = np.concatenate(OverallDataStructure['BhvFileName'], axis=0)
    DrugSessions = np.concatenate(OverallDataStructure['DrugDay'], axis=1)
    # NumberOfDrugTrPerSession
    NumberOfDrugTrPerSession = np.zeros(len(SessionWiseDMs))
    Subject1stInitial = []
    for i in range(0, len(SessionWiseDMs)):
        NumberOfDrugTrPerSession[i] = np.sum(
            (SessionWiseDMs[i]['NewVarForTrRef'].reshape(-1) == 2) * (SessionWiseDMs[i]['resp_trials']))
        Subject1stInitial.append(FileNameForSessions[i][13])

    HSessionIndices = [i for i, x in enumerate(Subject1stInitial) if x == "H"]
    ASessionIndices = [i for i, x in enumerate(Subject1stInitial) if x == "A"]

    MinHTrials = NumberOfDrugTrPerSession[HSessionIndices][DrugSessions[0][HSessionIndices] == 0].min()
    MinATrials = NumberOfDrugTrPerSession[ASessionIndices][DrugSessions[0][ASessionIndices] == 0].min()

    MinTrReq = np.zeros(len(NumberOfDrugTrPerSession))
    MinTrReq[HSessionIndices] = MinHTrials
    MinTrReq[ASessionIndices] = MinATrials

    KetamineSes = (NumberOfDrugTrPerSession >= MinTrReq) * DrugSessions[0]
    SalineSes = (NumberOfDrugTrPerSession >= MinTrReq) * (DrugSessions[0] == 0)
    return KetamineSes, SalineSes

def MakeFig8(BinnedAnalysis, SessionWiseDMs, KetamineSes, SalineSes, BootstrapNoIN, PermNoIn):
    ##Data for panel A

    t_list_Pcorr_RT = np.arange(-20, 61)
    Pcorr_t_mean_list_saline_avg = np.nanmean(BinnedAnalysis[SalineSes == 1, :], axis=0)
    Pcorr_t_se_list_saline_avg = np.nanstd(BinnedAnalysis[SalineSes == 1, :], axis=0, ddof=1) / np.sqrt(
        np.sum(SalineSes == 1))
    Pcorr_t_mean_list_ketamine_avg = np.nanmean(BinnedAnalysis[KetamineSes == 1, :], axis=0)
    Pcorr_t_se_list_ketamine_avg = np.nanstd(BinnedAnalysis[KetamineSes == 1, :], axis=0, ddof=1) / np.sqrt(
        np.sum(KetamineSes == 1))

    TrialEr_Ket = np.concatenate(
        [SessionWiseDMs[i]['CompTr_TrialError'] for i in np.argwhere(KetamineSes).reshape(-1)][:], axis=0)
    TrialEr_Sal = np.concatenate(
        [SessionWiseDMs[i]['CompTr_TrialError'] for i in np.argwhere(SalineSes).reshape(-1)][:], axis=0)
    EvidData_Ket = np.concatenate(
        [SessionWiseDMs[i]['OnDrugTemporalWeightsDM'] for i in np.argwhere(KetamineSes).reshape(-1)][:], axis=0)
    EvidData_Sal = np.concatenate(
        [SessionWiseDMs[i]['OnDrugTemporalWeightsDM'] for i in np.argwhere(SalineSes).reshape(-1)][:], axis=0)

    EvidData_Preket = np.concatenate(
        [SessionWiseDMs[i]['PreDrugTemporalWeightsDM'] for i in np.argwhere(KetamineSes).reshape(-1)][:], axis=0)
    EvidData_Presal = np.concatenate(
        [SessionWiseDMs[i]['PreDrugTemporalWeightsDM'] for i in np.argwhere(SalineSes).reshape(-1)][:], axis=0)

    ## Data for panels D to F
    Reg_bars_avg_ketamine, Reg_bars_err_avg_ketamine, Reg_bars_avg_saline, Reg_bars_err_avg_saline, Reg_bars_avg_pre_ketamine, Reg_bars_avg_pre_saline, PermutationpValsLogMod, ParamsToOutputKet, ParamsToOutputSal, \
    BootStrapDataKet, BootStrapDataSal, PermutationpVals = DrugDayPVBAnalysis(EvidData_Ket, EvidData_Sal,
                                                                              EvidData_Preket, EvidData_Presal,
                                                                              BootstrapNoIN, PermNoIn)

    # Re-organise the data neatly for plotting
    mean_effect_list_avg = np.array(
        [Reg_bars_avg_saline[1], Reg_bars_avg_ketamine[1]])  # Saline/ketamine. Mean Regressor
    var_effect_list_avg = np.array(
        [Reg_bars_avg_saline[2], Reg_bars_avg_ketamine[2]])  # Saline/ketamine. Variance Regressor
    var_mean_ratio_list_avg = var_effect_list_avg / mean_effect_list_avg  # Saline/ketamine. Variance Regressor/ Mean Regressor

    mean_effect_list_avg_preSK = np.array(
        [Reg_bars_avg_pre_saline[1], Reg_bars_avg_pre_ketamine[1]])  # Saline/ketamine. Mean Regressor
    var_effect_list_avg_preSK = np.array(
        [Reg_bars_avg_pre_saline[2], Reg_bars_avg_pre_ketamine[2]])  # Saline/ketamine. Variance Regressor
    var_mean_ratio_list_avg_preSK = var_effect_list_avg_preSK / mean_effect_list_avg_preSK  # Saline/ketamine. Variance Regressor/ Mean Regressor

    Mean_reg_err_bars_avg_v2 = np.abs([Reg_bars_err_avg_saline[1], Reg_bars_err_avg_ketamine[1]])
    Var_reg_err_bars_avg_v2 = np.abs([Reg_bars_err_avg_saline[2], Reg_bars_err_avg_ketamine[2]])
    Var_mean_ratio_err_Reg_bars_avg_v2 = var_mean_ratio_list_avg * (
            (Var_reg_err_bars_avg_v2 / var_effect_list_avg) ** 2 + (
            Mean_reg_err_bars_avg_v2 / mean_effect_list_avg) ** 2) ** 0.5

    ##Data for panel G
    betasKET, stdErsKET, betasSAL, stdErsSAL, ParamsToOutputKet, ParamsToOutputSal, BootStrapDataKet, BootStrapDataSal, PermutationpVals = \
        DrugDayPkAnalysis(EvidData_Ket, EvidData_Sal, BootstrapNoIN, PermNoIn)
    i_PK_list_6 = np.arange(1, 6 + 1)
    PK_avg_ketamine = betasKET[1:]
    PK_avg_saline = betasSAL[1:]
    PK_avg_ketamine_errbar = stdErsKET[1:]
    PK_avg_saline_errbar = stdErsSAL[1:]

    ## Fit psychometric functions
    dx_list, P_corr_Subj_list, ErrBar_P_corr_Subj_list, lik_model = \
        PsychometricFit(np.ones(len(EvidData_Ket)) == 1, EvidData_Ket[:, 0:6], EvidData_Ket[:, 6:-1],
                        2 - EvidData_Ket[:, -1], TrialEr_Ket, MethodInput='NarrowBroad')

    d_evidence_avg_ketamine_list = 100 * np.hstack((dx_list[1:8], dx_list[15:-1]))
    P_corr_avg_ketamine_list = np.hstack((P_corr_Subj_list[1:8], P_corr_Subj_list[15:-1]))
    ErrBar_P_corr_avg_ketamine_list = np.hstack((ErrBar_P_corr_Subj_list[0][1:8], ErrBar_P_corr_Subj_list[0][15:-1]))
    psychometric_params_avg_ketamine_all = lik_model.x

    dx_list, P_corr_Subj_list, ErrBar_P_corr_Subj_list, lik_model = \
        PsychometricFit(np.ones(len(EvidData_Sal)) == 1, EvidData_Sal[:, 0:6], EvidData_Sal[:, 6:-1],
                        2 - EvidData_Sal[:, -1], TrialEr_Sal, MethodInput='NarrowBroad')

    d_evidence_avg_saline_list = 100 * np.hstack((dx_list[1:8], dx_list[15:-1]))
    P_corr_avg_saline_list = np.hstack((P_corr_Subj_list[1:8], P_corr_Subj_list[15:-1]))
    ErrBar_P_corr_avg_saline_list = np.hstack((ErrBar_P_corr_Subj_list[0][1:8], ErrBar_P_corr_Subj_list[0][15:-1]))
    psychometric_params_avg_saline_all = lik_model.x

    # X-values to feed into the psychometric function for visualistion
    x_list_psychometric = np.arange(0.01, 0.5, 0.01)
    x0_psychometric = 0.

    ## ## Define subfigure domain.
    figsize = (max15, 0.8 * max15)
    width1_11 = 0.22
    width1_12 = 0.2
    width1_13 = width1_12
    width1_21 = 0.1
    width1_22 = width1_21
    width1_23 = width1_21
    width1_24 = width1_11
    x1_11 = 0.098
    x1_12 = x1_11 + width1_11 + 1.25 * xbuf0
    x1_13 = x1_12 + width1_12 + 1.15 * xbuf0
    x1_21 = 0.0825
    x1_22 = x1_21 + width1_21 + 0.8 * xbuf0
    x1_23 = x1_22 + width1_22 + 1.1 * xbuf0
    x1_24 = x1_23 + width1_23 + 1.55 * xbuf0  # x1_24 = x1_23 + width1_23 + 1.25*xbuf0
    height1_11 = 0.3
    height1_12 = height1_11
    height1_13 = height1_12
    height1_21 = 0.27
    height1_22 = height1_21
    height1_23 = height1_21
    height1_24 = height1_21
    y1_11 = 0.63
    y1_12 = y1_11
    y1_13 = y1_12
    y1_21 = y1_11 - height1_21 - 3.1 * ybuf0
    y1_22 = y1_21
    y1_23 = y1_22
    y1_24 = y1_23 + 0.23 * ybuf0

    rect1_11 = [x1_11, y1_11, width1_11, height1_11]
    rect1_12_0 = [x1_12, y1_12, width1_12 * 0.05, height1_12]
    rect1_12 = [x1_12 + width1_12 * 0.2, y1_12, width1_12 * (1 - 0.2), height1_12]
    rect1_13_0 = [x1_13, y1_13, width1_13 * 0.05, height1_13]
    rect1_13 = [x1_13 + width1_13 * 0.2, y1_13, width1_13 * (1 - 0.2), height1_13]
    rect1_21 = [x1_21, y1_21, width1_21, height1_21]
    rect1_22 = [x1_22, y1_22, width1_22, height1_22]
    rect1_23 = [x1_23, y1_23, width1_23, height1_23]
    rect1_24 = [x1_24, y1_24, width1_24, height1_24]

    ##### Plotting subfigure domain.
    fig_temp = plt.figure(num=8, figsize=figsize)
    fig_temp.text(0.005, 0.925, 'A', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.007 + x1_12 - x1_11, 0.925, 'B', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.007 + x1_13 - x1_11, 0.925, 'C', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.005, 0.975 + y1_21 - y1_11, 'D', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.047 + x1_22 - x1_21, 0.975 + y1_21 - y1_11, 'E', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(0.049 + x1_23 - x1_21, 0.975 + y1_21 - y1_11, 'F', fontsize=fontsize_fig_label, fontweight='bold')
    fig_temp.text(-0.001 + x1_24 - x1_21, 0.975 + y1_21 - y1_11, 'G', fontsize=fontsize_fig_label, fontweight='bold')
    bar_width_compare3 = 1.
    fig_temp.text(0.495, 0.95, 'Saline', fontsize=fontsize_fig_label, fontweight='bold', rotation='horizontal',
                  color='k')
    fig_temp.text(0.475 + x1_13 - x1_12, 0.95, 'Ketamine', fontsize=fontsize_fig_label, fontweight='bold',
                  rotation='horizontal', color='k')
    fig_temp.text(0.14 - x1_11 + x1_21, 0.933 + y1_21 - y1_11, 'Mean Evidence\nBeta', fontsize=fontsize_fig_label - 1,
                  rotation='horizontal', color='k', va='center', horizontalalignment='center')
    fig_temp.text(0.3305 - x1_11 + x1_21, 0.933 + y1_21 - y1_11, 'SD Evidence\nBeta', fontsize=fontsize_fig_label - 1,
                  rotation='horizontal', color='k', va='center', horizontalalignment='center')
    fig_temp.text(0.5535 - x1_11 + x1_21, 0.933 + y1_21 - y1_11, 'PVB Index', fontsize=fontsize_fig_label - 1,
                  rotation='horizontal', color='k', ha='center', va='center')

    ## rect1_11: Correct Probability vs time, Both Monkeys
    ax = fig_temp.add_axes(rect1_11)
    remove_topright_spines(ax)
    ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_avg, color=color_list_expt[0], linestyle='-', zorder=3,
            clip_on=False, label='Saline', linewidth=1.)  # , dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
    ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_avg + Pcorr_t_se_list_saline_avg, color=color_list_expt[0],
            linestyle='-', zorder=2, clip_on=False,
            linewidth=0.5)  # , dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
    ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_saline_avg - Pcorr_t_se_list_saline_avg, color=color_list_expt[0],
            linestyle='-', zorder=2, clip_on=False,
            linewidth=0.5)  # , dashes=(3.5,1.5))#, linestyle=linestyle_list[i_var_a])
    ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_avg, color=color_list_expt[1], linestyle='-', zorder=3,
            clip_on=False, label='Ketamine', linewidth=1.)  # , linestyle=linestyle_list[i_var_a])
    ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_avg + Pcorr_t_se_list_ketamine_avg, color=color_list_expt[1],
            linestyle='-', zorder=2, clip_on=False, linewidth=0.5)  # , linestyle=linestyle_list[i_var_a])
    ax.plot(t_list_Pcorr_RT, Pcorr_t_mean_list_ketamine_avg - Pcorr_t_se_list_ketamine_avg, color=color_list_expt[1],
            linestyle='-', zorder=2, clip_on=False, linewidth=0.5)  # , linestyle=linestyle_list[i_var_a])
    ax.fill_between([5., 30.], 1., lw=0, color='k', alpha=0.2, zorder=0)
    ax.set_xlabel('Time (mins)', fontsize=fontsize_legend, labelpad=1.)
    ax.set_ylabel('Correct Probability', fontsize=fontsize_legend, labelpad=2.)
    ax.set_xlim([-20, 60])
    ax.set_ylim([0.5, 1.])
    ax.set_xticks([-20, 0, 20, 40, 60])
    ax.set_yticks([0.5, 1.])
    ax.yaxis.set_ticklabels([0.5, 1])
    minorLocator = MultipleLocator(0.1)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    legend = ax.legend(loc=(0.39, -0.02), fontsize=fontsize_legend - 1, frameon=False, ncol=1, markerscale=-1.,
                       columnspacing=1., handletextpad=0.2)
    for color, text, item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
        text.set_color(color)
        item.set_visible(False)

    ## rect1_12: Psychometric function. Saline and ketamine
    AxesPos = [[rect1_12_0, rect1_12], [rect1_13_0, rect1_13]]
    ParamsHere = [psychometric_params_avg_saline_all, psychometric_params_avg_ketamine_all]
    DataHere = [[d_evidence_avg_saline_list, P_corr_avg_saline_list, ErrBar_P_corr_avg_saline_list], \
                [d_evidence_avg_ketamine_list, P_corr_avg_ketamine_list, ErrBar_P_corr_avg_ketamine_list]]
    LegLocs = [(-0.34, -0.12), (-0.6, 0.74)]  # legend location

    for ii in np.arange(2):  # loop across subplots w/ same code for saline and ketamine psychometrics

        ax_0 = fig_temp.add_axes(AxesPos[ii][0])
        ax = fig_temp.add_axes(AxesPos[ii][1])

        remove_topright_spines(ax_0)
        remove_topright_spines(ax)
        ax.spines['left'].set_visible(False)
        remove_topright_spines(ax)
        ax.errorbar(DataHere[ii][0][:], DataHere[ii][1][:], DataHere[ii][2][:],
                    color=color_list_expt[ii], ecolor=color_list_expt[ii], fmt='.', zorder=4, clip_on=False,
                    markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6,
                    capsize=1.)  # , linestyle=linestyle_list[i_var_a])
        ax.errorbar(-DataHere[ii][0][:], 1. - DataHere[ii][1][:], DataHere[ii][2][:],
                    color=[1 - (1 - ci) * 0.5 for ci in color_list_expt[ii]],
                    ecolor=[1 - (1 - ci) * 0.5 for ci in color_list_expt[ii]], fmt='.', zorder=4, clip_on=False,
                    markeredgecolor='k', linewidth=0.3, elinewidth=0.6, markeredgewidth=0.6,
                    capsize=1.)  # , linestyle=linestyle_list[i_var_a])
        ax.plot(100. * x_list_psychometric,
                PsychFitterNB(x_list_psychometric, ParamsHere[ii]), color=color_list_expt[ii],
                ls='-', clip_on=False, zorder=3, label='Higher SD Correct')  # , linestyle=linestyle_list[i_var_a])
        ax.plot(100. * x_list_psychometric,
                1. - PsychFitterNB(-x_list_psychometric, ParamsHere[ii]),
                color=[1 - (1 - ci) * 0.5 for ci in color_list_expt[ii]], ls='-', clip_on=False, zorder=2,
                label='Lower SD Correct')  # , linestyle=linestyle_list[i_var_a])
        ax_0.scatter(100. * x0_psychometric, PsychFitterNB(x0_psychometric, ParamsHere[ii]),
                     s=15., color=color_list_expt[ii], marker='_', clip_on=False,
                     linewidth=1.305)  # , linestyle=linestyle_list[i_var_a])
        ax_0.scatter(100. * x0_psychometric,
                     1. - PsychFitterNB(-x0_psychometric, ParamsHere[ii]), s=15.,
                     color=[1 - (1 - ci) * 0.5 for ci in color_list_expt[ii]], marker='_', clip_on=False,
                     linewidth=1.305)  # , linestyle=linestyle_list[i_var_a])
        ax.plot([0.3, 50], [0.5, 0.5], linewidth=0.7, color='k', ls='--', clip_on=False)
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
        ax_0.tick_params(direction='out', pad=1.5)
        ax_0.tick_params(which='minor', direction='out')
        ax.tick_params(direction='out', pad=1.5)
        ax.tick_params(which='minor', direction='out')
        ## Add breakmark = wiggle
        kwargs = dict(transform=ax_0.transAxes, color='k', linewidth=1, clip_on=False)
        y_shift_spines = -0.08864
        ax_0.plot((1, 1 + 2. / 3.), (y_shift_spines + 0., y_shift_spines + 0.05), **kwargs)  # top-left diagonal
        ax_0.plot((1 + 2. / 3., 1 + 4. / 3,), (y_shift_spines + 0.05, y_shift_spines - 0.05),
                  **kwargs)  # top-left diagonal
        ax_0.plot((1 + 4. / 3., 1 + 6. / 3.), (y_shift_spines - 0.05, y_shift_spines + 0.),
                  **kwargs)  # top-left diagonal
        ax_0.plot((1 + 6. / 3., 1 + 9. / 3.), (y_shift_spines + 0., y_shift_spines + 0.), **kwargs)  # top-left diagonal
        ax_0.spines['left'].set_position(('outward', 5))
        ax_0.spines['bottom'].set_position(('outward', 7))
        ax.spines['bottom'].set_position(('outward', 7))
        legend = ax.legend(loc=LegLocs[ii], fontsize=fontsize_legend - 1, frameon=False, ncol=1, markerscale=0.,
                           columnspacing=1., handletextpad=0., labelspacing=0.3)
        for color, text, item in zip([color_list_expt[ii], [1 - (1 - ci) * 0.5 for ci in color_list_expt[ii]]],
                                     legend.get_texts(), legend.legendHandles):
            text.set_color(color)
            item.set_visible(False)

    ## rect1_21: Mean Beta, Model and perturbations
    ax = fig_temp.add_axes(rect1_21)
    remove_topright_spines(ax)
    ax.bar(np.arange(len(mean_effect_list_avg)), mean_effect_list_avg, bar_width_compare3, alpha=bar_opacity,
           yerr=Mean_reg_err_bars_avg_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge',
           linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
    ax.plot([0, 2. * bar_width_compare3], [0.5 * (mean_effect_list_avg_preSK[0] + mean_effect_list_avg_preSK[1]),
                                           0.5 * (mean_effect_list_avg_preSK[0] + mean_effect_list_avg_preSK[1])],
            ls='--', color='k', clip_on=False, lw=0.8)  # Pre saline/ketamine values
    ax.scatter([1.], [25.2], s=16., color='k', marker=(5, 2), clip_on=False,
               zorder=10)  # , linestyle=linestyle_list[i_var_a])
    ax.plot([0.5, 1.5], [24., 24.], ls='-', lw=1., color='k', clip_on=False,
            zorder=9)  # , linestyle=linestyle_list[i_var_a])
    ax.set_xlim([0, len(mean_effect_list_avg) - 1 + bar_width_compare3])
    ax.set_ylim([0., 26.])
    ax.set_xticks([0., 1.])
    ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
    ax.set_yticks([0., 25.])
    ax.set_yticklabels([0., 0.25])
    minorLocator = MultipleLocator(5.)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=0.)
    ax.tick_params(which='minor', direction='out')
    ax.tick_params(bottom="off")

    ## rect1_22: Variance Beta, Model and perturbations
    ax = fig_temp.add_axes(rect1_22)
    remove_topright_spines(ax)
    ax.bar(np.arange(len(var_effect_list_avg)), var_effect_list_avg, bar_width_compare3, alpha=bar_opacity,
           yerr=Var_reg_err_bars_avg_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False, align='edge',
           linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
    ax.plot([0, 2. * bar_width_compare3], [0.5 * (var_effect_list_avg_preSK[0] + var_effect_list_avg_preSK[1]),
                                           0.5 * (var_effect_list_avg_preSK[0] + var_effect_list_avg_preSK[1])],
            ls='--', color='k', clip_on=False, lw=0.8)  # Pre saline/ketamine values
    ax.set_xlim([0, len(var_effect_list_avg) - 1 + bar_width_compare3])
    ax.set_ylim([0., 5.])
    ax.set_xticks([0., 1.])
    ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
    ax.set_yticks([0., 5.])
    ax.set_yticklabels([0., 0.05])
    minorLocator = MultipleLocator(1.)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=0.)
    ax.tick_params(which='minor', direction='out')
    ax.tick_params(bottom="off")

    ## rect1_23: Variance Beta/ Mean Beta, Model and perturbations
    ax = fig_temp.add_axes(rect1_23)
    remove_topright_spines(ax)
    ax.bar(np.arange(len(var_mean_ratio_list_avg)), var_mean_ratio_list_avg, bar_width_compare3, alpha=bar_opacity,
           yerr=Var_mean_ratio_err_Reg_bars_avg_v2, ecolor='k', color=color_list_expt, edgecolor='k', clip_on=False,
           align='edge', linewidth=1., error_kw=dict(elinewidth=0.8, markeredgewidth=0.8), capsize=2.)
    ax.plot([0, 2. * bar_width_compare3], [0.5 * (var_mean_ratio_list_avg_preSK[0] + var_mean_ratio_list_avg_preSK[1]),
                                           0.5 * (var_mean_ratio_list_avg_preSK[0] + var_mean_ratio_list_avg_preSK[1])],
            ls='--', color='k', clip_on=False, lw=0.8)  # Pre saline/ketamine values
    ax.scatter([1.], [0.49], s=16., color='k', marker=(5, 2), clip_on=False,
               zorder=10)  # , linestyle=linestyle_list[i_var_a])
    ax.plot([0.5, 1.5], [0.46, 0.46], ls='-', lw=1., color='k', clip_on=False,
            zorder=9)  # , linestyle=linestyle_list[i_var_a])
    ax.set_xlim([0, len(var_mean_ratio_list_avg) - 1 + bar_width_compare3])
    ax.set_ylim([0., 0.5])
    ax.set_xticks([0., 1.])
    ax.xaxis.set_ticklabels(['Saline', 'Ketamine'], rotation=30)
    ax.set_yticks([0., 0.5])
    ax.yaxis.set_ticklabels([0, 0.5])
    minorLocator = MultipleLocator(0.1)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=0.)
    ax.tick_params(which='minor', direction='out')
    ax.tick_params(bottom="off")

    ## rect1_24: Psychophysical Kernel, panel G
    ax = fig_temp.add_axes(rect1_24)
    remove_topright_spines(ax)
    tmp = ax.errorbar(i_PK_list_6, PK_avg_ketamine, PK_avg_ketamine_errbar, color=color_list_expt[1], linestyle='-',
                      marker='.', zorder=(3 - 1), clip_on=False, markeredgecolor='k', elinewidth=0.6,
                      markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    tmp = ax.errorbar(i_PK_list_6, PK_avg_saline, PK_avg_saline_errbar, color=color_list_expt[0], linestyle='-',
                      marker='.', zorder=(3 - 1), clip_on=False, markeredgecolor='k', elinewidth=0.6,
                      markeredgewidth=0.6, capsize=1.)  # , linestyle=linestyle_list[i_var_a])
    for b in tmp[1]:
        b.set_clip_on(False)
    ax.set_xlabel('Sample Number', fontsize=fontsize_legend)
    ax.set_ylabel('Stimuli Beta', fontsize=fontsize_legend)
    ax.set_ylim([0., 4.35])
    ax.set_xlim([1, 6])
    ax.set_xticks([1, 6])
    ax.set_yticks([0., 4.])
    ax.text(0.1, 4.5, r'$\times\mathregular{10^{-2}}$', fontsize=fontsize_tick)
    minorLocator = MultipleLocator(1.)
    ax.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(1.)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction='out', pad=1.5)
    ax.tick_params(which='minor', direction='out')
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.plot(i_PK_list_6, PK_avg_saline, label='Saline', color=color_list_expt[0], linestyle='-', zorder=0,
            clip_on=False)  # , linestyle=linestyle_list[i_var_a])
    ax.plot(i_PK_list_6, PK_avg_ketamine, label='Ketamine', color=color_list_expt[1], linestyle='-', zorder=0,
            clip_on=False)  # , linestyle=linestyle_list[i_var_a])
    legend = ax.legend(loc=(0., 0.4), fontsize=fontsize_legend - 1, frameon=False, ncol=1, markerscale=0.,
                       columnspacing=1., handletextpad=0.)
    for color, text, item in zip(color_list_expt, legend.get_texts(), legend.legendHandles):
        text.set_color(color)
        item.set_visible(False)
    fig_temp.savefig(FIGUREFileLocations + 'Figure8_2022.pdf')  # Finally save fig
    print('     Completed Figure 8')
    return

def DrugDayPVBAnalysis(EvidData_Ket, EvidData_Sal, EvidData_Preket, EvidData_Presal, BootstrapNoIN, PermNoIN):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H%M%S")
    NewFileNameToSaveUnder = DATAFileLocations + "PVBLapseData" + dt_string + ".h5"
    NAMEOFACTUALDATATOLOAD = DATAFileLocations + 'PVBLapseData30_01_2023_215353.h5'  # python generated data
    NAMEOFACTUALDATATOLOAD = DATAFileLocations + 'PVBLapseDataFromMatlab.h5'  # data used in publication

    ##Fit a linear model of PVB to explain choices
    DM_ket = np.vstack((np.ones((1, len(EvidData_Ket)))
                        , np.mean(EvidData_Ket[:, 6:-1], axis=1) - np.mean(EvidData_Ket[:, :6], axis=1), \
                        np.std(EvidData_Ket[:, 6:-1], axis=1, ddof=1) - np.std(EvidData_Ket[:, :6], axis=1, ddof=1))).T
    Y_ket = EvidData_Ket[:, -1] == 2
    m = sm.Logit(Y_ket, DM_ket).fit(disp=0)
    KetLogParams, KetLogErs = m.params, m.bse

    DM_sal = np.vstack((np.ones((1, len(EvidData_Sal)))
                        , np.mean(EvidData_Sal[:, 6:-1], axis=1) - np.mean(EvidData_Sal[:, :6], axis=1), \
                        np.std(EvidData_Sal[:, 6:-1], axis=1, ddof=1) - np.std(EvidData_Sal[:, :6], axis=1, ddof=1))).T
    Y_sal = EvidData_Sal[:, -1] == 2
    m = sm.Logit(Y_sal, DM_sal).fit(disp=0)
    SalLogParams, SalLogErs = m.params, m.bse

    DM_preket = np.vstack((np.ones((1, len(EvidData_Preket)))
                           , np.mean(EvidData_Preket[:, 6:-1], axis=1) - np.mean(EvidData_Preket[:, :6], axis=1), \
                           np.std(EvidData_Preket[:, 6:-1], axis=1, ddof=1) - np.std(EvidData_Preket[:, :6], axis=1,
                                                                                     ddof=1))).T
    Y_preket = EvidData_Preket[:, -1] == 2
    m = sm.Logit(Y_preket, DM_preket).fit(disp=0)
    PreKetLogParams, PreKetLogErs = m.params, m.bse

    DM_presal = np.vstack((np.ones((1, len(EvidData_Presal)))
                           , np.mean(EvidData_Presal[:, 6:-1], axis=1) - np.mean(EvidData_Presal[:, :6], axis=1), \
                           np.std(EvidData_Presal[:, 6:-1], axis=1, ddof=1) - np.std(EvidData_Presal[:, :6], axis=1,
                                                                                     ddof=1))).T
    Y_presal = EvidData_Presal[:, -1] == 2
    m = sm.Logit(Y_presal, DM_presal).fit(disp=0)
    PreSalLogParams, PreSalLogErs = m.params, m.bse

    ## Run a permutation on the above logistic model to compare ketamine and saline
    TruePVB_Ket = KetLogParams[2] / KetLogParams[1]
    TruePVB_Sal = SalLogParams[2] / SalLogParams[1]
    TruePVBDif = TruePVB_Ket - TruePVB_Sal
    TrueMeanBetaDif = KetLogParams[1] - SalLogParams[1]
    TrueSdBetaDif = KetLogParams[2] - SalLogParams[2]

    PermNoLogModel = 1000  # for speed, was 1,000,000 in the paper
    CombY = np.hstack((Y_ket, Y_sal))
    CombDM = np.vstack((DM_ket, DM_sal))
    nTrSal = len(Y_sal)
    nTrKet = len(Y_ket)

    p_perm_mean_beta = 0
    p_perm_SD_beta = 0
    p_perm_pvbindex = 0

    for i in range(0, PermNoLogModel):
        RandomTrOrder = np.random.permutation(nTrSal + nTrKet)
        PermKetMod = sm.Logit(CombY[RandomTrOrder[:nTrKet]], CombDM[RandomTrOrder[:nTrKet], :]).fit(disp=0)
        PermSalMod = sm.Logit(CombY[RandomTrOrder[nTrKet:]], CombDM[RandomTrOrder[nTrKet:], :]).fit(disp=0)

        PermPVBDif = (PermKetMod.params[2] / PermKetMod.params[1]) - (PermSalMod.params[2] / PermSalMod.params[1])
        PermMeanBetaDif = PermKetMod.params[1] - PermSalMod.params[1]
        PermSdBetaDif = PermKetMod.params[2] - PermSalMod.params[2]

        if abs(PermPVBDif) > abs(TruePVBDif):
            p_perm_pvbindex = p_perm_pvbindex + 1

        if abs(PermMeanBetaDif) > abs(TrueMeanBetaDif):
            p_perm_mean_beta = p_perm_mean_beta + 1

        if abs(PermSdBetaDif) > abs(TrueSdBetaDif):
            p_perm_SD_beta = p_perm_SD_beta + 1

        if np.remainder(100 * (i + 1) / PermNoLogModel, 10) == 0:
            print('     ' + str(100 * (i + 1) / PermNoLogModel) + ' percent through PVB logistic permutatation test')

    p_perm_mean_beta = p_perm_mean_beta / PermNoLogModel
    p_perm_SD_beta = p_perm_SD_beta / PermNoLogModel
    p_perm_pvbindex = p_perm_pvbindex / PermNoLogModel
    PermutationpValsLogMod = [p_perm_mean_beta, p_perm_SD_beta, p_perm_pvbindex]

    ## Fit the PVB mode for ketamine and saline with a lapse term

    if BootstrapNoIN > 0:
        # Fit a model w/ lapse terms and get error estimates using bootstrapping
        ParamsToOutputKet, BootStrapDataKet = FitRegWithALapseTerm(np.column_stack((DM_ket[:, 1:], Y_ket)),
                                                                   BootstrapNo=BootstrapNoIN)
        ParamsToOutputSal, BootStrapDataSal = FitRegWithALapseTerm(np.column_stack((DM_sal[:, 1:], Y_sal)),
                                                                   BootstrapNo=BootstrapNoIN)

        hf = h5py.File(NewFileNameToSaveUnder, 'a')
        hf['/Ketamine/ParamsToOutputKet'] = ParamsToOutputKet
        hf['/Ketamine/BootStrapDataKet'] = BootStrapDataKet
        hf['/Ketamine'].attrs['TimeAnalysisPerformed'] = dt_string

        hf['/Saline/ParamsToOutputSal'] = ParamsToOutputSal
        hf['/Saline/BootStrapDataSal'] = BootStrapDataSal
        hf['/Saline'].attrs['TimeAnalysisPerformed'] = dt_string
        hf.close()
    else:
        hf_in = h5py.File(NAMEOFACTUALDATATOLOAD, 'r')
        # list(hf_in.keys())
        ParamsToOutputKet = np.array(hf_in['Ketamine/ParamsToOutputKet'])
        ParamsToOutputSal = np.array(hf_in['Saline/ParamsToOutputSal'])
        BootStrapDataKet = np.array(hf_in['Ketamine/BootStrapDataKet'])
        BootStrapDataSal = np.array(hf_in['Saline/BootStrapDataSal'])
        hf_in.close()
        print('     Loaded pre-existing matlab data for PVB permutatation test with lapse term')

    # Permutation test with a lapse term
    if PermNoIN > 0:
        # FitRegWithALapseTermPermTest
        PermNullMatrix = []
        ParamsToOutputKet_true = FitRegWithALapseTerm(np.column_stack((DM_ket[:, 1:], Y_ket)), BootstrapNo=0)
        ParamsToOutputSal_true = FitRegWithALapseTerm(np.column_stack((DM_sal[:, 1:], Y_sal)), BootstrapNo=0)
        TrueDiff = abs(ParamsToOutputKet_true - ParamsToOutputSal_true)
        TrueDiff = np.hstack((TrueDiff, abs((ParamsToOutputKet_true[3] / ParamsToOutputKet_true[2]) - (
                    ParamsToOutputSal_true[3] / ParamsToOutputSal_true[2]))))

        nSalTr = len(DM_sal)
        nKetTr = len(DM_ket)
        DM_combined = np.vstack([DM_ket, DM_sal])
        Y_combined = np.hstack([Y_ket, Y_sal])

        for i in range(0, PermNoIN):
            PermutedTrOrder = np.random.permutation(np.arange(nSalTr + nKetTr))
            DM_Ketperm = DM_combined[PermutedTrOrder[:nKetTr], :]
            Y_Ketperm = Y_combined[PermutedTrOrder[:nKetTr]]
            DM_Salperm = DM_combined[PermutedTrOrder[nKetTr:], :]
            Y_Salperm = Y_combined[PermutedTrOrder[nKetTr:]]

            ParamsToOutputKet_perm = FitRegWithALapseTerm(np.column_stack((DM_Ketperm[:, 1:], Y_Ketperm)),
                                                          BootstrapNo=0)
            ParamsToOutputSal_perm = FitRegWithALapseTerm(np.column_stack((DM_Salperm[:, 1:], Y_Salperm)),
                                                          BootstrapNo=0)
            PermNullMatrix_I = np.hstack((ParamsToOutputKet_perm - ParamsToOutputSal_perm,
                                          abs((ParamsToOutputKet_perm[3] / ParamsToOutputKet_perm[2]) - (
                                                      ParamsToOutputSal_perm[3] / ParamsToOutputSal_perm[2]))))

            PermNullMatrix.append(PermNullMatrix_I)

        PermutationpVals = np.sum(TrueDiff < abs(np.array(PermNullMatrix)), axis=0) / PermNoIN

        hf = h5py.File(NewFileNameToSaveUnder, 'a')
        hf['/Ketamine_v_saline/PermutationpVals'] = PermutationpVals
        hf['/Ketamine_v_saline'].attrs['TimeAnalysisPerformed'] = dt_string
        hf.close()

    else:
        hf_in = h5py.File(NAMEOFACTUALDATATOLOAD, 'r')
        # list(hf_in.keys())
        PermutationpVals = np.array(hf_in['Ketamine_v_saline/PermutationpVals'])
        hf_in.close()

    return KetLogParams, KetLogErs, SalLogParams, SalLogErs, PreKetLogParams, PreSalLogParams, PermutationpValsLogMod, ParamsToOutputKet, ParamsToOutputSal, \
           BootStrapDataKet, BootStrapDataSal, PermutationpVals


def DrugDayPkAnalysis(EvidData_Ket, EvidData_Sal, BootstrapNoIN, PermNoIN):
    ## Set up a file name to save any new data produced from bootsrapping / permutaton testing
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H%M%S")
    NewFileNameToSaveUnder = DATAFileLocations + "PKLapseData" + dt_string + ".h5"
    NAMEOFACTUALDATATOLOAD = DATAFileLocations + 'PKLapseData30_01_2023_215353.h5'  # python generated data
    NAMEOFACTUALDATATOLOAD = DATAFileLocations + 'PKLapseDataFromMatlab.h5'  # data used in publication

    ## Reformat the data to create design matricies
    DM_Ket = np.concatenate([np.ones((len(EvidData_Ket), 1)), EvidData_Ket[:, 0:6] - EvidData_Ket[:, 6:-1]], axis=1)
    DM_Sal = np.concatenate([np.ones((len(EvidData_Sal), 1)), EvidData_Sal[:, 0:6] - EvidData_Sal[:, 6:-1]], axis=1)
    Y_Ket = EvidData_Ket[:, -1] == 1
    Y_Sal = EvidData_Sal[:, -1] == 1

    ## Use a simple logistic model to calculate parameters
    m = sm.Logit(Y_Ket, DM_Ket).fit(disp=0)
    betasKET, pvals, tvals, stdErsKET = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics
    m = sm.Logit(Y_Sal, DM_Sal).fit(disp=0)
    betasSAL, pvals, tvals, stdErsSAL = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    if BootstrapNoIN > 0:
        # Fit a model w/ lapse terms and get error estimates using bootstrapping
        ParamsToOutputKet, BootStrapDataKet = FitRegWithALapseTerm(np.column_stack((DM_Ket[:, 1:], Y_Ket)),
                                                                   BootstrapNo=BootstrapNoIN)
        ParamsToOutputSal, BootStrapDataSal = FitRegWithALapseTerm(np.column_stack((DM_Sal[:, 1:], Y_Sal)),
                                                                   BootstrapNo=BootstrapNoIN)

        hf = h5py.File(NewFileNameToSaveUnder, 'a')
        hf['/Ketamine/ParamsToOutputKet'] = ParamsToOutputKet
        hf['/Ketamine/BootStrapDataKet'] = BootStrapDataKet
        hf['/Ketamine'].attrs['TimeAnalysisPerformed'] = dt_string

        hf['/Saline/ParamsToOutputSal'] = ParamsToOutputSal
        hf['/Saline/BootStrapDataSal'] = BootStrapDataSal
        hf['/Saline'].attrs['TimeAnalysisPerformed'] = dt_string
        hf.close()
    else:
        hf_in = h5py.File(NAMEOFACTUALDATATOLOAD, 'r')
        # list(hf_in.keys())
        ParamsToOutputKet = np.array(hf_in['Ketamine/ParamsToOutputKet'])
        ParamsToOutputSal = np.array(hf_in['Saline/ParamsToOutputSal'])
        BootStrapDataKet = np.array(hf_in['Ketamine/BootStrapDataKet'])
        BootStrapDataSal = np.array(hf_in['Saline/BootStrapDataSal'])
        hf_in.close()
        print('     Loaded pre-existing matlab data for PK permutatation test with lapse term')


    if PermNoIN > 0:
        # FitRegWithALapseTermPermTest
        PermNullMatrix = []
        ParamsToOutputKet_true = FitRegWithALapseTerm(np.column_stack((DM_Ket[:, 1:], Y_Ket)), BootstrapNo=0)
        ParamsToOutputSal_true = FitRegWithALapseTerm(np.column_stack((DM_Sal[:, 1:], Y_Sal)), BootstrapNo=0)
        TrueDiff = abs(ParamsToOutputKet_true - ParamsToOutputSal_true)

        nSalTr = len(DM_Sal)
        nKetTr = len(DM_Ket)
        DM_combined = np.vstack([DM_Ket, DM_Sal])
        Y_combined = np.hstack([Y_Ket, Y_Sal])

        for i in range(0, PermNoIN):
            PermutedTrOrder = np.random.permutation(np.arange(nSalTr + nKetTr))
            DM_Ketperm = DM_combined[PermutedTrOrder[:nKetTr], :]
            Y_Ketperm = Y_combined[PermutedTrOrder[:nKetTr]]
            DM_Salperm = DM_combined[PermutedTrOrder[nKetTr:], :]
            Y_Salperm = Y_combined[PermutedTrOrder[nKetTr:]]

            ParamsToOutputKet_perm = FitRegWithALapseTerm(np.column_stack((DM_Ketperm[:, 1:], Y_Ketperm)),
                                                          BootstrapNo=0)
            ParamsToOutputSal_perm = FitRegWithALapseTerm(np.column_stack((DM_Salperm[:, 1:], Y_Salperm)),
                                                          BootstrapNo=0)
            PermNullMatrix.append(ParamsToOutputKet_perm - ParamsToOutputSal_perm)

        PermutationpVals = np.sum(TrueDiff < abs(np.array(PermNullMatrix)), axis=0) / PermNoIN

        hf = h5py.File(NewFileNameToSaveUnder, 'a')
        hf['/Ketamine_v_saline/PermutationpVals'] = PermutationpVals
        hf['/Ketamine_v_saline'].attrs['TimeAnalysisPerformed'] = dt_string
        hf.close()

    else:
        hf_in = h5py.File(NAMEOFACTUALDATATOLOAD, 'r')
        # list(hf_in.keys())
        PermutationpVals = np.array(hf_in['Ketamine_v_saline/PermutationpVals'])
        hf_in.close()

    return betasKET, stdErsKET, betasSAL, stdErsSAL, ParamsToOutputKet, ParamsToOutputSal, \
           BootStrapDataKet, BootStrapDataSal, PermutationpVals


def FitRegWithALapseTerm(Input_DM_y, BootstrapNo):
    # Description of input variables:
    # DM_Input - nTr x nSamples + 1: each column is a regressor of interest, with a final column appended for whether the left option was chosen
    # BootstrapNo - number of runs needed to create errorbars. Set to 0 if not required.
    DM_True = np.concatenate([np.ones((len(Input_DM_y), 1)), Input_DM_y[:, :-1]], axis=1)
    Y_True = Input_DM_y[:, -1] == 1

    ## Use a simple logistic model to calculate parameters
    m = sm.Logit(Y_True, DM_True).fit(disp=0)
    betas, pvals, tvals, stdErs = m.params, m.pvalues, m.tvalues, m.bse  # Extract our statistics

    ## Use a more complex model which incorporates a lapse term
    StartingParams = np.append(np.array(0.1), m.params)

    def LOG_withlapse(params):
        preds = params[0] + (1 - (2 * params[0])) / (1 + np.exp(-np.matmul(DM_Here, params[1:])))
        Neg_LLofModel = -np.sum(np.log((Y_Here * preds) + ((1 - Y_Here) * (1 - preds))))
        Lambda = 0.01
        err = Neg_LLofModel + np.sum(np.power(params, 2)) * Lambda
        if sum(preds > 1) > 0 or sum(preds < 0) > 0 or params[0] < 0 or params[0] > 1:
            err = 1000000000000000000000000000000000000000000000
        return err

    DM_Here = DM_True
    Y_Here = Y_True
    lik_model = minimize(LOG_withlapse, StartingParams)
    paramstooutput = lik_model.x

    ## Use a bootsrap process to generate errorbars for the above parameters
    StoreOfBootstrapParams = []
    if BootstrapNo > 0:
        print('Beginning a bootstrap procedure with ' + str(BootstrapNo) + ' Iterations')
        for Bi in range(0, BootstrapNo):
            TrInds = choices(range(0, len(DM_Here)), k=len(DM_Here))
            m = sm.Logit(Y_True[TrInds], DM_True[TrInds, :]).fit(disp=0)
            StartingParamsBoot = np.append(np.array(0.1), m.params)
            DM_Here = DM_True[TrInds, :]
            Y_Here = Y_True[TrInds]

            def LOG_withlapseBOOT(params):
                preds = params[0] + (1 - (2 * params[0])) / (1 + np.exp(-np.matmul(DM_Here, params[1:])))
                Neg_LLofModel = -np.sum(np.log((Y_Here * preds) + ((1 - Y_Here) * (1 - preds))))
                Lambda = 0.01
                err = Neg_LLofModel + np.sum(np.power(params, 2)) * Lambda
                if sum(preds > 1) > 0 or sum(preds < 0) > 0 or params[0] < 0 or params[0] > 1:
                    err = 1000000000000000000000000000000000000000000000
                return err

            lik_model = minimize(LOG_withlapseBOOT, StartingParamsBoot)
            paramstooutputBOOT = lik_model.x
            StoreOfBootstrapParams.append(paramstooutputBOOT)
        return paramstooutput, np.vstack(StoreOfBootstrapParams[:])
    else:
        return paramstooutput
