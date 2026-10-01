import numpy as np
from wdam.utils import config
from wdam.data.generator import generate
from wdam.data.windows import prepare,bounds
from wdam.baselines import simple,Ridge
from wdam.metrics import metrics,seasonal_scale


def test_generator_and_shift_pairing():
    c=config('data_train.yaml'); a=generate(c,0); b=generate(c,0); d=generate(c,1)
    for k in a: np.testing.assert_array_equal(a[k],b[k])
    assert not np.array_equal(a['y'],d['y'])
    shifted=generate({**c,'spike_median':c['spike_median']*2},0)
    np.testing.assert_allclose(shifted['spike'],a['spike']*2)
    np.testing.assert_array_equal(shifted['noise'],a['noise'])


def test_no_leakage():
    data=generate(config('data_train.yaml'),0); parts,s=prepare(data)
    train_end=bounds(len(data['y']))['train'][1]
    assert s.mean==data['y'][:train_end].mean()
    assert parts['val']['idx'].min()-parts['train']['t'].max()>24
    assert parts['test']['idx'].min()-parts['val']['t'].max()>24
    for w in parts.values():
        np.testing.assert_array_equal(w['idx'][:,-1]+1,w['t'])
        np.testing.assert_array_equal(w['idx'][:,0]+24,w['t'])
    modified={**data,'y':data['y'].copy()}; modified['y'][train_end:]=1e9
    _,new=prepare(modified)
    assert new==s


def test_baselines_and_metrics():
    x=np.arange(24,dtype=float)[None,:,None]
    pred=simple(x)
    assert pred['last_value'][0]==23 and pred['moving_average'][0]==11.5 and pred['seasonal_naive'][0]==0
    m=metrics([1.,3.],[2.,1.],2.)
    assert m['mae']==1.5 and m['rmse']==np.sqrt(2.5) and m['mase']==.75
    assert seasonal_scale(np.arange(30))==24
    assert metrics([],[],0)['mae'] is None


def test_ridge_fits_constant():
    w={'x':np.ones((40,24,5)),'t':np.arange(24,64),'y':np.full(40,3.)}
    r=Ridge().fit(w)
    np.testing.assert_allclose(r.predict(w),3.,atol=1e-10)
