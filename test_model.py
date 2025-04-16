import SequenceLearningDynamics.model as mo
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


if __name__=='__main__':
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
