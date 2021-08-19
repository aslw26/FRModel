import pytest

from tests.base.D2Fixture.test_fixture import f, c


def test_slice(f, c):
    """ Tests the NumPy Slicing for the XY Dims """
    f = f[10:20, 15:25]
    assert 300 == f.size()

def test_slice_w_obj(f, c):
    """ Tests the NumPy Slicing for the XY Dims """
    s1 = slice(10, 20, None)
    s2 = slice(15, 25, None)
    assert 300 == f[s1, s2].size()

def test_channel_index(f, c):
    """ Tests indexing a channel """
    assert 1, f[0, 0, c.RED].size()

def test_channel_multi_index(f, c):
    """ Tests indexing multiple channels """
    f_ = f[0, 0, [c.RED, c.BLUE]]
    assert 2 == f_.size()
    assert (1, 1, 2) == f_.shape

def test_duplicate_channel(f, c):
    """ Tests if we can duplicate a channel """
    with pytest.raises(KeyError):
        _ = f[0, 0, [c.RED, c.RED]]

def test_step(f, c):
    """ Tests slicing with stepping of 2 """
    assert 75 == f[0:10:2, 10:20:2].size()

def test_ellipsis(f, c):
    """ Tests ellipsis for xy with channel indexing """
    assert len(f[..., c.RED].channels) == 1

def test_bad_type(f, c):
    """ Tests rejection of Channel Bad Type """
    with pytest.raises(KeyError):
        _ = f[0, 0, 0]

def test_bad_channel_slice(f, c):
    """ Tests rejection of Channel Bad Type """
    with pytest.raises(KeyError):
        _ = f[0, 0, c.RED:c.GREEN]

def test_bad_channel_type_slice(f, c):
    """ Tests rejection of Channel Type Slicing """
    with pytest.raises(KeyError):
        _ = f[0, 0, 0:1]

def test_bad_absent_slice(f, c):
    """ Tests rejection of Channel Slicing """
    with pytest.raises(KeyError):
        _ = f[0, 0, c.NDVI]
