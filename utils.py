import pandas as pd
import numpy as np
from typing import List

from tqdm import tqdm
import setglobals as gl
import seaborn as sns
from matplotlib import ticker
from statsmodels.stats.anova import AnovaRM

total_sub_num = 16
seq_length  = 5


def read_dat_file(filename: str):
    data = pd.read_csv(filename, delimiter='\t')
    return data


def read_dat_files_subjs_list(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    data = [read_dat_file(gl.data_dir + "SL3_s" + f'{sub:02}' + ".dat") for sub in subjs_list]
    for sub in subjs_list:
        data[sub - 1]['SubNum'] = sub
    return data



def remove_no_go_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes no-go trials
    """

    return subj[subj['announce'] == 0]


def select_training_trials(subjs: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the training trials
    """

    return subjs[subjs['trialType'] == 2]



def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'RT'+str(i+1)
        col2 = 'RT'+str(i+2)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    subj['IPI0'] = subj['RT1']



def add_seq_pressed(subj: pd.DataFrame):
    """
    Adds the sequence pressed by the subject to the dataframe
    """

    subj['seqPressed'] = subj['resp1'].astype(str) + subj['resp2'].astype(str) + subj['resp3'].astype(str) + subj['resp4'].astype(str) + subj['resp5'].astype(str)





def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """


    subj_melted = pd.melt(subj,
                    id_vars=['BN', 'TN', 'SubNum', 'seqType', 'board', 'day', 'trialPoints', 'latePress',
                              'hardPress', 'seqError'],
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number',
                    value_name='IPI_Value')


    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)




    return subj_melted



def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj,
                    id_vars=['BN', 'TN', 'SubNum', 'seqType', 'board', 'day', 'trialPoints', 'latePress',
                              'hardPress', 'seqError'],
                    value_vars =  [_ for _ in subj.columns if _.startswith('resp')],
                    var_name='Response_Number',
                    value_name='Response_Value')

    subj_melted['N'] = subj_melted['Response_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_responses = finger_melt_responses(subj)

    merged_df = melt_IPIs.merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'seqType',
                                                      'board', 'day', 'trialPoints',
                                                      'latePress','hardPress', 'seqError', 'N'] )

    return merged_df



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['trialPoints'] >= 0)]


def finger_melt_Forces(subjs_force: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each Finger Force in the whole experiment adding two columns, "Force_Number" determining the order of Force
    and "Force_Value" determining the time of Force
    """


    subj_force_melted = pd.melt(subjs_force,
                    id_vars=['state', 'timeReal', 'time','BN', 'TN', 'SubNum', 'seqType',
                                                      'board', 'day', 'trialPoints',
                                                      'latePress','hardPress', 'seqError', 'IPI0', 'MT'],
                    value_vars =  [_ for _ in subjs_force.columns if _.startswith('force')],
                    var_name='Force_Number',
                    value_name='Force_Value')

    return subj_force_melted



def cut_force(subjs_force: pd.DataFrame, side_padding) -> pd.DataFrame:
    """
    Cuts the force data to the same length as the IPI data
    """
    subjs_force = subjs_force[(subjs_force['IPI0'] <= subjs_force['time'] + side_padding) & (subjs_force['time'] <= subjs_force['IPI0'] + subjs_force['MT'] + side_padding)]
    return subjs_force



def cut_force_left(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[(subjs_force['IPI0'] >= subjs_force['time'])]
    return subjs_force


def cut_force_right(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[(subjs_force['IPI0'] + subjs_force['MT'] <= subjs_force['time'])]
    return subjs_force

def kernel_smoother(x, y, z, xi, yi, h = 0.1):
    """
    smooth scattered (x,y,z) onto grid (xi, yi) using Gaussian kernel smoother
    """

    # query points flattened
    xq = np.column_stack((xi.flatten(), yi.flatten()))
    x = np.column_stack((x, y))

    zout = np.zeros(len(xq))

    for j, q in enumerate(xq):
        # distances from query point to all data points
        d = np.linalg.norm(x - q, axis=1)
        # Gaussian kernel weights
        w = np.exp(-(d/h)**2 / 2)
        #normalized weighted average
        if np.sum(w) > 0:
            zout[j] = np.sum(w * z) / np.sum(w)
        else:
            zout[j] = np.nan
    return zout.reshape(xi.shape)


def kernel_smoother_1d(x, y, xi, h = 0.1):
    """
    smooth scattered (x,y) onto grid (xi) using Gaussian kernel smoother
    """
    xq = xi.flatten()
    zout = np.zeros(len(xq))

    for j, q in enumerate(xq):
        # distances from query point to all data points
        d = np.abs(x - q)
        # Gaussian kernel weights
        w = np.exp(-(d/h)**2 / 2)
        #normalized weighted average
        if np.sum(w) > 0:
            zout[j] = np.sum(w * y) / np.sum(w)
        else:
            zout[j] = np.nan
    return zout.reshape(xi.shape)

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

def set_figure_style(scale="1col"):
    """
    Set figure styling based on publication constraints.
    
    Parameters:
        scale (str): Scale of the figure, choose from "1col", "1.5col", "2col".
                     - "1col" for 8.5cm
                     - "1.5col" for 11.6cm
                     - "2col" for 17.6cm
    """
    # Define width options in cm
    widths = {"1col": 7.62, "1.5col": 11.6, "2col": 16.5}
    
    if scale not in widths:
        raise ValueError("Invalid scale. Choose from '1col', '1.5col', or '2col'.")
    
    # Convert width from cm to inches (1 cm = 0.393701 inches)
    width_in = widths[scale] * 0.393701
    
    # Set figure size (width, height)
    # Assuming height proportional to width (Golden Ratio)
    golden_ratio = (5**0.5 - 1) / 2
    rcParams["figure.figsize"] = (width_in, width_in * golden_ratio)
    
    # Set font sizes
    rcParams["font.size"] = 10  # General font size
    # rcParams["font.size"] = 20  # General font size
    rcParams["axes.titlesize"] = 12  # Figure title
    # rcParams["axes.titlesize"] = 26  # Figure title
    rcParams["axes.labelsize"] = 9  # Axis main label
    # rcParams["axes.labelsize"] = 22  # Axis main label
    rcParams["xtick.labelsize"] = 7  # Tick labels
    # rcParams["xtick.labelsize"] = 16  # Tick labels
    rcParams["ytick.labelsize"] = 7
    # rcParams["ytick.labelsize"] = 16
    rcParams["legend.fontsize"] = 8  # Legend entries
    # rcParams["legend.fontsize"] = 20  # Legend entries
    rcParams["figure.titleweight"] = "bold"
    
    # Set stroke width
    rcParams["axes.linewidth"] = 0.75
    # rcParams["axes.linewidth"] = 1.5

    # rcParams["lines.linewidth"] = 3
    
    rcParams["xtick.major.width"] = 0.75
    # rcParams["xtick.major.width"] = 1.5
    rcParams["ytick.major.width"] = 0.75
    # rcParams["ytick.major.width"] = 1.5

    
    # Subpanel lettering size
    rcParams["text.usetex"] = False  # Set to True if using LaTeX
    rcParams["axes.formatter.use_mathtext"] = True  # Math text for scientific notation

def add_subpanel_label(ax, label, fontsize=20, position=(-0.1, 1.05)):
    """
    Add a subpanel label (e.g., 'a', 'b') to a subplot.
    
    Parameters:
        ax (Axes): Matplotlib Axes object.
        label (str): The label text.
        fontsize (int): Font size for the label.
        position (tuple): Position of the label in axes coordinates.
    """
    ax.text(position[0], position[1], label, transform=ax.transAxes, 
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


### FORCE DISTANCE MATRIX PLOTTING
### for a given subject, plot the force distance matrix across all trials
### error trials are marked on x and y ticks
### red dashed lines separate days
### Execution times aligned to the bottom x-axis

def plot_force_movement_dynamics(data, subj, n_trials_per_day, n_days):

    subdata = data[data['SubNum'] == subj]
    force_vectors = np.stack(subdata['forceVector'].values)
    diff = force_vectors[:, np.newaxis, :] - force_vectors[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    vmin, vmax = np.percentile(distances, [5, 95])  # better visualization

    # Create subplots: top for heatmap, bottom for execution times
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [6, 1]})

    # Heatmap on top
    sns.heatmap(distances, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax1, cbar = False)
    for day in range(1, n_days):
        ax1.axvline(n_trials_per_day * day, color='red', linestyle='--', linewidth=0.7)
        ax1.axhline(n_trials_per_day * day, color='red', linestyle='--', linewidth=0.7)

    # Add colorbar to the right of the heatmap
    cbar_ax = fig.add_axes([1, 0.2, 0.01, 0.7])
    fig.colorbar(ax1.collections[0], cax=cbar_ax)

    # Plotting error trials on x and y ticks for heatmap
    error_trials = subdata[subdata['isError'] == 1]['T'] - 1  # zero-indexed
    ax1.set_xticks(error_trials)
    ax1.set_yticks(error_trials)
    ax1.set_xticklabels([''] * len(error_trials), color='black')  # Hide labels but keep ticks
    ax1.set_yticklabels([''] * len(error_trials), color='black')
    ax1.set_title(f'Subject {subj} Force Distance Matrix')
    ax1.set_xlabel('')  # Remove x-label for now
    ax1.set_ylabel('Trial Number')

    # Bottom plot for execution times
    trial_numbers = subdata['T'] - 1  # zero-indexed
    execution_times = subdata['ET']
    ax2.plot(trial_numbers, execution_times, color='black', linewidth=1)
    ax2.set_xlim(ax1.get_xlim())  # Align x-axis with heatmap
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('ET')

    # Add vertical lines for days on bottom plot
    for day in range(1, n_days):
        ax2.axvline(n_trials_per_day * day, color='red', linestyle='--', linewidth=0.7)

    # Mark error trials on bottom plot
    ax2.scatter(error_trials, execution_times.iloc[error_trials], color='red', s=2)

    sns.despine()

    plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots for alignment
    plt.show()


def calc_neighbour_distances(data):
    distances = []
    data_correct = data[data['points'] != -1]

    for subind, subdata in tqdm(data_correct.groupby('SubNum')):
        for block, block_data in subdata.groupby('BN'):
            mean_block_et = block_data['ET'].mean()
            for i in range(len(block_data) - 1):
                trial1 = block_data.iloc[i]
                trial2 = block_data.iloc[i + 1]
                if trial2['T'] - trial1['T'] == 1:  # only consider consecutive trials
                    # Calculate force distance
                    force_dist = np.linalg.norm(trial2['forceVector'] - trial1['forceVector'])
                    # Calculate ETs
                    first_ET = trial1['ET'] - mean_block_et
                    second_ET = trial2['ET'] - mean_block_et

                    distances.append({
                        'SubNum': subind,
                        'Force_Distance': force_dist,
                        'first_ET': first_ET,
                        'second_ET': second_ET
                    })

    # Concatenate all distances into a single DataFrame
    distances = pd.DataFrame(distances)
    distances['count'] = 1
    return distances





def plot_neighbour_distances(distances):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[6, 1, 5])
    
    ax_dens = fig.add_subplot(gs[0, 0])
    ax_cont_top = fig.add_subplot(gs[0, 1])
    ax_cont_bot = fig.add_subplot(gs[1, 1])
    ax_point = fig.add_subplot(gs[2, 0])

    # 1. Density Plot
    cmap = sns.color_palette('Blues', as_cmap=True)
    df_f = distances[(distances['first_ET'].between(-500, 500)) & (distances['second_ET'].between(-500, 500))]
    sns.kdeplot(data=df_f, x='first_ET', y='second_ET', fill=True, cmap=cmap, cbar=True, ax=ax_dens)
    ax_dens.plot([-500, 500], [-500, 500], color='black', linestyle='--', alpha=0.5)
    ax_dens.set_xlabel(r'$E_T -\overline{E}$ (ms)')
    ax_dens.set_ylabel(r'$E_{T+1} -\overline{E}$ (ms)')
    ax_dens.set_title('Density Plot')

    # 2. Contour Plot
    force_cutoff = df_f['Force_Distance'].quantile(0.95)
    df_cont = df_f[df_f['Force_Distance'] <= force_cutoff]
    
    xi = np.linspace(df_cont['first_ET'].min(), df_cont['first_ET'].max(), 100)
    yi = np.linspace(df_cont['second_ET'].min(), df_cont['second_ET'].max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = kernel_smoother(df_cont['first_ET'], df_cont['second_ET'], df_cont['Force_Distance'], XI, YI, h=50)
    
    m1 = ax_cont_top.contourf(XI, YI, ZI, cmap=cmap, levels=20)
    fig.colorbar(m1, ax=ax_cont_top)
    ax_cont_top.plot([-500, 500], [-500, 500], 'k--', alpha=0.5)
    ax_cont_top.set_title(r'|| $\vec{F}_{t+1} - \vec{F}_{t}$ ||')
    ax_cont_top.set_ylabel(r'$E_{T+1} -\overline{E}$ (ms)')

    # Marginal Contour
    xi_m, yi_m = np.meshgrid(xi, [0, 2])
    zi_m = kernel_smoother_1d(df_cont['first_ET'], df_cont['Force_Distance'], xi_m, h=50)
    m2 = ax_cont_bot.contourf(xi_m, yi_m, zi_m, cmap=cmap, levels=20)
    cbar2 = fig.colorbar(m2, ax=ax_cont_bot, aspect=3)
    cbar2.locator = ticker.MaxNLocator(nbins=3)
    ax_cont_bot.set_yticks([])
    ax_cont_bot.set_xlabel(r'$E_T -\overline{E}$ (ms)')
    ax_cont_bot.set_ylabel('marginal')

    # 3. Summary Pointplot
    ET_diff_bin_size = 200
    distances['abs_ET_diff'] = np.abs(distances['second_ET'] - distances['first_ET'])
    df_p = distances[distances['abs_ET_diff'] < 400].copy()
    df_p['is_pos_diff'] = (df_p['second_ET'] - df_p['first_ET']) > 0
    df_p['ET_diff_bin'] = df_p['abs_ET_diff'] // ET_diff_bin_size + 1
    
    grouped = df_p.groupby(['SubNum', 'ET_diff_bin', 'is_pos_diff']).agg({'Force_Distance': 'median'}).reset_index()
    sns.pointplot(data=grouped, x='ET_diff_bin', y='Force_Distance', hue='is_pos_diff', 
                  hue_order=[True, False], palette='colorblind', errorbar='se', ax=ax_point)
    
    ax_point.set_xlabel(r'|| $E_{t+1} - E_{t}$ ||')
    ax_point.set_ylabel(r'|| $\vec{F}_{t+1} - \vec{F}_{t}$ ||')
    handles, labels = ax_point.get_legend_handles_labels()
    ax_point.legend(handles, [r'$E_{t+1} > E_{t}$', r'$E_{t+1} < E_{t}$'], title='')
    
    bin_labels = [f'{(b-1)*ET_diff_bin_size}-{b*ET_diff_bin_size}' for b in sorted(grouped['ET_diff_bin'].unique())]
    ax_point.set_xticks(range(len(bin_labels)))
    ax_point.set_xticklabels(bin_labels)

    sns.despine(trim=True)
    plt.show()
    print("ANOVA results for neighbour distances:")
    print(AnovaRM(grouped, 'Force_Distance', 'SubNum', within=['ET_diff_bin', 'is_pos_diff']).fit())


def calc_triplet_distances(data):
    distances = []

    for subind, subdata in tqdm(data.groupby('SubNum')):
        for block, block_data in subdata.groupby('BN'):
            mean_block_et = block_data['ET'].mean()
            for i in range(len(block_data) - 2):
                trial1 = block_data.iloc[i]
                trial2 = block_data.iloc[i + 1]
                trial3 = block_data.iloc[i + 2]
                if (trial2['T'] - trial1['T'] == 1) and (trial3['T'] - trial2['T'] == 1):  # only consider triplets of consecutive trials
                    if (trial1['points'] != -1) and (trial3['points'] != -1): # if the first and last are correct
                        # Calculate force distance
                        force_dist = np.linalg.norm(trial3['forceVector'] - trial1['forceVector'])
                        # Calculate ET distance
                        et_dist = trial3['ET'] - trial1['ET']
                        error_in_the_middle = (trial2['points'] == -1)

                        distances.append({
                            'SubNum': subind,
                            'Force_Distance': force_dist,
                            'ET_Distance': np.abs(et_dist),
                            'is_middle_error': error_in_the_middle 
                        })

    # Concatenate all distances into a single DataFrame
    distances = pd.DataFrame(distances)
    distances['count'] = 1
    return distances


def plot_triplet_distances(distances):
    ET_diff_bin_size = 100
    filtered_distances = distances[distances['ET_Distance'] < 200].copy()
    filtered_distances['ET_diff_bin'] = filtered_distances['ET_Distance'] // ET_diff_bin_size + 1
    grouped_distances = filtered_distances.groupby(['SubNum', 'ET_diff_bin', 'is_middle_error']).agg({
        'Force_Distance': 'median',
        'count': 'sum'
    }).reset_index()

    # Fix hue order so legend ordering is stable
    hue_order = [False, True]
    ax = sns.pointplot(
        data=grouped_distances,
        x='ET_diff_bin',
        y='Force_Distance',
        errorbar='se',
        hue = 'is_middle_error',
        hue_order=hue_order,
        palette='colorblind'
    )
    plt.xlabel(r' $|E_{t+1} - E_{t-1}|$')
    plt.ylabel(r'|| $\vec{F}_{t+1} - \vec{F}_{t-1}$ ||')

    handles, labels = ax.get_legend_handles_labels()
    label_map = {'True': 't is error', 'False': 't is correct'}
    ax.legend(handles, [label_map[l] for l in labels], title='', loc='upper right', bbox_to_anchor=(1.3, 1))


    # Create bin labels showing the actual intervals
    bin_labels = []
    for bin_num in sorted(grouped_distances['ET_diff_bin'].unique()):
        lower = (bin_num - 1) * ET_diff_bin_size
        upper = bin_num * ET_diff_bin_size
        bin_labels.append(f'{lower}-{upper}')
    plt.xticks(range(len(bin_labels)), bin_labels)

    sns.despine(trim=True)

    print("ANOVA results for for triplet distances::")
    print(AnovaRM(grouped_distances, 'Force_Distance', 'SubNum', within=['ET_diff_bin', 'is_middle_error']).fit())
