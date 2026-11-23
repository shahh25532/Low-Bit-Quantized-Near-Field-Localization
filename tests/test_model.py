import torch
import pytest
from nearfield_ext.model import HybridLocalizer,SubarrayHybridCombiner

@pytest.mark.parametrize('arch',['cnn','mlp'])
@pytest.mark.parametrize('bits',[None,1,2,4])
def test_model_forward_backward(arch,bits):
    torch.manual_seed(11)
    model=HybridLocalizer(16,4,arch,bits,3.0)
    x=torch.randn(5,32)
    outputs,clip=model.forward_with_diagnostics(x)
    assert outputs.shape==(5,2)
    assert 0<=float(clip)<=1
    outputs.square().mean().backward()
    assert model.combiner.phases.grad is not None
    assert torch.isfinite(model.combiner.phases.grad).all()


def test_subarray_partition():
    with pytest.raises(ValueError): SubarrayHybridCombiner(10,3)
