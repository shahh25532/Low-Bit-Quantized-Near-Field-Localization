import numpy as np
import pytest
from nearfield_ext.simulation import SimulationConfig,make_splits,near_field_array_response,denormalize_target


def test_no_position_leakage_and_per_snr_balance():
    cfg=SimulationConfig(antennas=16,samples_per_snr=30,snr_db=(-10,0,10),seed=22)
    d=make_splits(cfg)
    ids={key:set(val['position_id']) for key,val in d.items()}
    assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    assert sum(len(v['x']) for v in d.values())==3*30
    for block in d.values():
        counts=np.unique(block['snr_db'],return_counts=True)[1]
        assert len(set(counts))==1
        assert block['x'].shape[1]==32


def test_reproducibility():
    cfg=SimulationConfig(antennas=8,samples_per_snr=20,snr_db=(-5,5))
    x=make_splits(cfg);y=make_splits(cfg)
    assert np.array_equal(x['test']['x'],y['test']['x'])


def test_array_response_unit_modulus():
    v=near_field_array_response(np.array([-0.5,0.2]),np.array([2.0,5.0]),8,25e9)
    assert v.shape==(2,8)
    assert np.allclose(np.abs(v),1)


def test_denormalization_geometry():
    xy=denormalize_target(np.array([[1.,-1.],[-1.,1.],[0.,0.]]),(1,10))
    assert np.allclose(xy[0],[1,0],atol=1e-6)
    assert np.allclose(xy[1],[-10,0],atol=1e-6)
    assert np.allclose(xy[2],[0,5.5],atol=1e-6)


def test_bad_config():
    with pytest.raises(ValueError): make_splits(SimulationConfig(antennas=3))
