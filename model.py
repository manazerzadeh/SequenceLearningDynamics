import numpy as np

class LdModel():
    def __init__(self):
        self.Y0 = 2500
        self.fast_tau = 1
        self.fast_B = 200
        self.slow_amp = 1300
        self.slow_B = 20
        self.slow_A = self.slow_amp/(self.slow_B+self.slow_amp)
        self.fatigue_tau = 0.3
        self.fatigue_B = 10
        self.param_list = ['fast_A','fast_tau','slow_amp','slow_tau','fatigue_tau','fatigue_A']

    def set_params(self,theta):
        self.nparams = len(self.param_list)
        if theta.shape[0]!=self.nparams:
            raise(NameError('theta not the right size'))
        for i,l in enumerate(self.param_list):
            self[l]=theta[i]
        return self

    def get_params(self):
        self.nparams = len(self.param_list)
        theta = np.zeros(self.nparams)
        for i,l in enumerate(param_list):
            theta[i] = self[l]
        return theta

class LdModel_additive(LdModel):

    def predict(self,time,seq=None):
        N = time.shape[0]
        z_fast = np.zeros((N))
        z_slow = np.zeros((N))
        z_fatigue = np.zeros((N))
        yp = np.zeros((N))
        for i,t in enumerate(time):
            yp[i]=self.Y0 - z_fast[i] - z_slow[i]  + z_fatigue[i]
            if i < N-1:
                dt = time[i+1]-time[i]
                fast_A =np.exp(-self.fast_tau*dt)
                z_fast[i+1] =  fast_A * (z_fast[i] +  self.fast_B)
                z_slow[i+1] = self.slow_A * (z_slow[i] + self.slow_B)
                fatigue_A =np.exp(-self.fatigue_tau*dt)
                z_fatigue[i+1] =  fatigue_A * (z_fatigue[i] +  self.fatigue_B)
        return yp, z_fast , z_slow,z_fatigue


class LdModel_additive(LdModel):

    def predict(self,time,seq=None):
        N = time.shape[0]
        z_fast = np.zeros((N))
        z_slow = np.zeros((N))
        z_fatigue = np.zeros((N))
        yp = np.zeros((N))
        for i,t in enumerate(time):
            yp[i]=self.Y0 - z_fast[i] - z_slow[i]  + z_fatigue[i]
            if i < N-1:
                dt = time[i+1]-time[i]
                fast_A =np.exp(-self.fast_tau*dt)
                z_fast[i+1] =  fast_A * (z_fast[i] +  self.fast_B)
                z_slow[i+1] = self.slow_A * (z_slow[i] + self.slow_B)
                fatigue_A =np.exp(-self.fatigue_tau*dt)
                z_fatigue[i+1] =  fatigue_A * (z_fatigue[i] +  self.fatigue_B)
        return yp, z_fast , z_slow,z_fatigue
