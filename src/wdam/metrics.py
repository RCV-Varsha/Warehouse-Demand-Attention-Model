import numpy as np


def seasonal_scale(train,lag=24):
    return float(np.mean(np.abs(np.asarray(train)[lag:]-np.asarray(train)[:-lag])))


def metrics(actual,predicted,denominator):
    actual,predicted=np.asarray(actual),np.asarray(predicted)
    assert actual.shape==predicted.shape
    if not len(actual): return dict(count=0,mae=None,rmse=None,mase=None,bias=None)
    e=predicted-actual
    mae=float(np.mean(np.abs(e)))
    return dict(count=len(e),mae=mae,rmse=float(np.sqrt(np.mean(e**2))),
                mase=mae/denominator if denominator>0 else None,bias=float(e.mean()))


def sliced(actual,pred,denominator,masks):
    return {'all':metrics(actual,pred,denominator),**{k:metrics(actual[v],pred[v],denominator) for k,v in masks.items()}}
