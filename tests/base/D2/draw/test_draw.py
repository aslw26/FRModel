import numpy as np

from frmodel.base.D2.draw2D import Draw2D

from tests.base.D2Fixture.test_fixture import fs as f


def test_draw_multiple(f):
    draw = Draw2D.load_frame(f)
    x = np.linspace(0, f.width(), 200)
    y = np.random.randint(0, f.height(), 200)
    draw.mark_multiple(x,
                       y,
                       outline=(255, 0, 0),
                       labels=[f"{i:.0f}, {j:.0f}" for i, j in zip(x, y)])
    # draw.save("./DrawMultipleRandom.png")

def test_draw_single(f):
    draw = Draw2D.load_frame(f)
    draw.mark_single(f.width() // 2,
                     f.height() // 2,
                     outline=(255, 0, 0),
                     label="Mid Point")
    # draw.save("./DrawSingle.png")

