import SequenceLearningDynamics.model as mo
import SequenceLearningDynamics.utils as ut
import SequenceLearningDynamics.setglobals as gl
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


n_trials_per_block = 20
n_trials_per_day = 400
n_blocks_per_day = 20
n_days = 3


def test_model():
    M = mo.LdModel_additive()
    time = np.concatenate([np.linspace(0,100,100),np.linspace(150,250,100),np.linspace(300,400,100)],axis=0)
    trial = np.arange(np.shape(time)[0])
    yp,z_fast,z_slow,z_fatigue = M.predict(time)
    plt.subplot(4,1,1)
    plt.plot(time,yp,'r.')
    plt.subplot(4,1,2)
    plt.plot(time,z_fast,'r.')
    plt.subplot(4,1,3)
    plt.plot(time,z_slow,'r.')
    plt.subplot(4,1,4)
    plt.plot(time,z_fatigue,'r.')
    pass


def get_data(subj) -> pd.DataFrame:
    """
    Gets the average data for a subject
    """
    D=pd.read_csv(gl.result_dir + 'SL3_all_trial_data.tsv', delimiter='\t')
    D['error'] = (D.seqError>0).astype(int)
    D['speed'] = 1/D.MT*1000*5

    if subj == 'all':
        D = D[D.error==0].groupby(['day','BN','TN']).agg({'speed':'mean','MT':'median','error':'min'}).reset_index()
    else:
        D = D[D.SubNum==subj]
        D[D.error==1].speed = np.nan
        D[D.MT==1].speed = np.nan
    D['tn'] = np.arange(D.shape[0])+1
    D['BN'] = D['BN'] -6
    D['BN_in_day'] = (D.BN  - (D.day-1)*n_blocks_per_day)
    D['time'] = D['day']*24*60*60/4 + (D['BN_in_day']-1)*35 + (D['TN']-1)
    return D

def plot_data(D):
    """
    Plots the data
    """
    plt.plot(D.tn, D.speed, 'k.')
    block = np.where(D.TN==2)[0]+1
    dblock = np.where((D.BN_in_day==1) & (D.TN==2))[0]+1
    for i in block:
        plt.axvline(x=i, color=(0.8,0.8,0.8), linestyle=':')
    for i in dblock:
        plt.axvline(x=i, color='k', linestyle='-')
    if 'yp' in D.columns:
        plt.plot(D.tn,D.yp,':')
    if 'z_slow' in D.columns:
        plt.plot(D.tn,D.z_slow,'b:')

def plot_block(data):
    D = data[data.error==0].groupby(['day','BN','BN_in_day']).agg({'speed':'mean','MT':'median','yp':'mean','z_slow':'mean'}).reset_index()
    plt.plot(D.BN, D.speed, 'k.')
    dblock = np.where((D.BN_in_day==1))[0]+0.5
    for i in dblock:
        plt.axvline(x=i, color='k', linestyle='-')
    plt.plot(D.BN,D.yp,':')
    plt.plot(D.BN,D.z_slow,'b:')

def plot_trial(D):
    D['BN_in_day'] = (D.BN  - (D.day-1)*n_blocks_per_day - 6)
    T=D[(D.error==0) & (D.day>1)]
    Dtrial = T.groupby(['TN']).agg({'speed':'mean','MT':'median','yp':'mean','z_slow':'mean'}).reset_index()
    plt.plot(Dtrial.TN, Dtrial.speed, 'k.')
    plt.plot(Dtrial.TN,Dtrial.yp,':')

if __name__=='__main__':
    D= get_data('all')
    M = mo.LdModel_additive()
    M = mo.fit_model(D,M)
    yp,z_fast,z_slow,z_fatigue = M.predict(D.time)
    z_slow = M.y0 + z_slow
    D['yp'] = yp
    D['z_slow'] = z_slow
    plt.figure(1)
    plot_data(D)
    plt.figure(2)
    plot_block(D)
    plt.figure(3)
    plot_trial(D)

    pass