import pytest

from tests.base.D2Fixture.test_fixture import f, c


def test_labels(f, c):
    """ Verify labels are correctly initialized"""
    assert 0 == f.labels[c.RED]
    assert 1 == f.labels[c.GREEN]
    assert 2 == f.labels[c.BLUE]

def test_data_present(f, c):
    """ Slices the data that is already present """
    assert 3 == f[..., c.RGB].shape[-1]
    assert 1 == f[..., c.RED].shape[-1]
    assert 1 == f[..., c.GREEN].shape[-1]
    assert 1 == f[..., c.BLUE].shape[-1]

def test_data_absent(f, c):
    """ Attempts to get data that isn't there. Will throw an exception """
    with pytest.raises(KeyError):
        _ = f[..., c.EX_G]

def test_get_present(f, c):
    """ Gets the channels that is already present. Synonymous with data_chn """
    assert 0 == f.get_chns(self_=False, chns=[c.RED])  .labels[c.RED]
    assert 0 == f.get_chns(self_=False, chns=[c.GREEN]).labels[c.GREEN]
    assert 0 == f.get_chns(self_=False, chns=[c.BLUE]) .labels[c.BLUE]

def test_get_calculate(f, c):
    """ Calculates channels that are absent """
    g = f.get_chns(self_=False, chns=[c.HSV])  # This should give us the HSV Channels only

    assert 0 == g[..., c.HUE].labels[c.HUE]
    assert 0 == g[..., c.SATURATION].labels[c.SATURATION]
    assert 0 == g[..., c.VALUE].labels[c.VALUE]

    # Red isn't present anymore due to self_=False
    with pytest.raises(KeyError):
        _ = g[..., c.RED]

def test_shape(f):
    """ Tests the data shape and orientation """
    assert (f.height(), f.width(), len(f.labels)) == f.shape

