import numpy as np


def simple(x):
    return dict(last_value=x[:,-1,0],moving_average=x[:,:,0].mean(1),seasonal_naive=x[:,0,0])


def ridge_features(window):
    t=window['t']; h=t%24; d=(t//24)%7
    return np.column_stack([window['x'][:,:,0],np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),
                            np.sin(2*np.pi*d/7),np.cos(2*np.pi*d/7)]).astype('float64')


class Ridge:
    def fit(self,window,alpha=10.):
        x=ridge_features(window); self.mean=x.mean(0); self.std=np.maximum(x.std(0),1e-8)
        x=np.column_stack([np.ones(len(x)),(x-self.mean)/self.std])
        penalty=np.eye(x.shape[1])*alpha; penalty[0,0]=0
        self.coef=np.linalg.solve(x.T@x+penalty,x.T@window['y'])
        return self

    def predict(self,window):
        x=(ridge_features(window)-self.mean)/self.std
        return np.column_stack([np.ones(len(x)),x])@self.coef
