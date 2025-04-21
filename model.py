import numpy as np
from scipy.optimize import minimize

class LdModel():
    def __init__(self):
        self.y0 = 2.2
        self.fast_tau = 0.01
        self.fast_B = 200
        self.slow_amp = 3
        self.slow_B = 2
        self.slow_A = self.slow_amp/(self.slow_B+self.slow_amp)
        self.fatigue_tau = 0.3
        self.fatigue_B = 0
        self.param_list = ['y0','fast_tau','slow_amp','slow_tau','fatigue_tau','fatigue_A']

    def update_params(self):
        self.fast_A = self.fast_amp/(self.fast_B+self.fast_amp)
        self.fast_tau = -np.log(self.fast_A )
        self.slow_A = self.slow_amp/(self.slow_B+self.slow_amp)

    def set_params(self,theta):
        self.nparams = len(self.param_list)
        if theta.shape[0]!=self.nparams:
            raise(NameError('theta not the right size'))
        for i,l in enumerate(self.param_list):
            setattr(self,l,np.exp(theta[i]))
        self.update_params()
        return self

    def get_params(self):
        self.nparams = len(self.param_list)
        theta = np.zeros(self.nparams)
        for i,l in enumerate(self.param_list):
            theta[i] = np.log(getattr(self,l))
        return theta

class LdModel_additive(LdModel):
    def __init__(self):
        self.y0 = 2
        self.fast_amp = 1.5
        self.fast_B = 0.04
        self.slow_amp = 3
        self.slow_B = 0.01
        self.fatigue_tau = 0.3
        self.fatigue_B = 0
        self.update_params()
        self.param_list = ['y0','fast_amp','fast_B','slow_amp','slow_B']

    def update_params(self):
        self.fast_A = self.fast_amp/(self.fast_B+self.fast_amp)
        self.fast_tau = -np.log(self.fast_A )
        self.slow_A = self.slow_amp/(self.slow_B+self.slow_amp)

    def predict(self,time,seq=None):
        N = time.shape[0]
        z_fast = np.zeros((N))
        z_slow = np.zeros((N))
        z_fatigue = np.zeros((N))
        yp = np.zeros((N))
        for i,t in enumerate(time):
            yp[i]=self.y0 + z_fast[i] + z_slow[i] - z_fatigue[i]
            if i < N-1:
                dt = time[i+1]-time[i]
                fast_A =np.exp(-self.fast_tau*dt)
                z_fast[i+1] =  fast_A * (z_fast[i] +  self.fast_B)
                z_slow[i+1] = self.slow_A * (z_slow[i] + self.slow_B)
                fatigue_A =np.exp(-self.fatigue_tau*dt)
                z_fatigue[i+1] =  fatigue_A * (z_fatigue[i] +  self.fatigue_B)
        return yp, z_fast , z_slow,z_fatigue

def L2_loss(theta,M,D):
    """
    Computes the L2 loss
    """
    M.set_params(theta)
    yp,z_fast,z_slow,z_fatigue = M.predict(D.time)
    loss = np.sum((D.speed-yp)**2)
    return loss

def fit_model(D, M):
    """
    Fits the model to the data
    """
    theta0=M.get_params()
    res = minimize(L2_loss, theta0, args=(M,D), method='L-BFGS-B')
    M=M.set_params(res.x)
    return M