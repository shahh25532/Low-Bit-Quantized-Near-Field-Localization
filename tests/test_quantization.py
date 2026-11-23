import pytest
import torch
from nearfield_ext.quantization import UniformADC


def test_float_is_identity():
    x=torch.tensor([-5.,-1.,0.,1.,5.])
    assert torch.equal(UniformADC(None)(x),x)

@pytest.mark.parametrize('bits', [1,2,4])
def test_level_count_saturation_and_midrise(bits):
    q=UniformADC(bits,full_scale=2.0)
    q.eval()
    x=torch.linspace(-10,10,1001)
    out=q(x)
    assert torch.unique(out).numel()==2**bits
    assert out.max()<2 and out.min()>-2
    assert out[0]<0 and out[-1]>0


def test_ste_passes_gradient():
    x=torch.tensor([-1.0,0.3,2.0],requires_grad=True)
    q=UniformADC(2,full_scale=3.)
    q.train()
    q(x).sum().backward()
    assert torch.allclose(x.grad,torch.ones_like(x))


def test_clipping_metric():
    q=UniformADC(1,full_scale=2.)
    assert q.clipping_fraction(torch.tensor([-4.,0.,1.,3.])).item()==0.5


def test_invalid_configs():
    with pytest.raises(ValueError): UniformADC(0)
    with pytest.raises(ValueError): UniformADC(4,full_scale=0)
