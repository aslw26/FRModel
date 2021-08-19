import pytest

from tests.base.D2Fixture.test_fixture import fs as f, c


def test_labels(f, c):
    """ Verify labels are correctly initialized"""
    assert 0 == f.labels[c.RED]
    assert 1 == f.labels[c.GREEN]
    assert 2 == f.labels[c.BLUE]
    assert 3 == f.labels[c.RED_EDGE]
    assert 4 == f.labels[c.NIR]

def test_get_present(f, c):
    """ Gets the channels that is already present. Synonymous with data_chn """
    assert 0 == f[..., c.RED]     .labels[c.RED]
    assert 0 == f[..., c.GREEN]   .labels[c.GREEN]
    assert 0 == f[..., c.BLUE]    .labels[c.BLUE]
    assert 0 == f[..., c.RED_EDGE].labels[c.RED_EDGE]
    assert 0 == f[..., c.NIR]     .labels[c.NIR]

def test_get_calculate(f, c):
    """ Calculates channels that are absent """
    g = f.get_chns(self_=False, chns=[c.NDVI])  # This should give us the HSV Channels only

    assert 0 == g[..., c.NDVI].labels[c.NDVI]

    # NIR isn't present anymore due to self_=False
    with pytest.raises(KeyError):
        _ = g[..., c.NIR]

def test_shape(f, c):
    """ Tests the data shape and orientation """
    assert (f.height(), f.width(), len(f.labels)) == f.shape
