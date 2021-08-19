from tests.base.D2Fixture.test_fixture import f, c


def test_plot(f, c):
    fc = f.get_chns(self_=False, chns=[f.CHN.XY, f.CHN.HSV])

    fpl = fc.plot()
    ROWS = 3
    COLS = 2
    PLT_SCALE = 1.1
    fpl.subplot_shape = (ROWS, COLS)
    fpl.image(scale=PLT_SCALE)


