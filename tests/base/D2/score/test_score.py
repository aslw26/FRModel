import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

from frmodel.base.D2.kmeans2D import KMeans2D
from tests.base.D2Fixture.test_fixture import frame_box as f, c


def test_box(f, c):

    frame_xy = f.get_chns(self_=False,
                          chns=[c.XY, c.HSV, c.MEX_G, c.EX_GR, c.NDI])

    km = KMeans2D(frame_xy,
                  KMeans(n_clusters=3, verbose=False),
                  fit_to=[c.MEX_G, c.EX_GR, c.NDI],
                  scaler=minmax_scale)
    kmf = km.as_frame()
    score = kmf.score(f)
    assert score['Custom'] == pytest.approx(1)
    assert score['Homogeneity'] == pytest.approx(1)
    assert score['Completeness'] == pytest.approx(1)
    assert score['V Measure'] == pytest.approx(1)

