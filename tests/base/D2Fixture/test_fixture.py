import pytest

from frmodel.base.D2 import Frame2D
from tests.path import RSC_PATH

# Arrange
@pytest.fixture
def frame_box():
    return Frame2D.from_image(RSC_PATH + f"imgs/rgb/test/box.png")

# Arrange
@pytest.fixture
def f():
    return Frame2D.from_image(RSC_PATH + f"imgs/rgb/test/sample.jpg")

# Arrange
@pytest.fixture
def fs():
    return Frame2D.load(RSC_PATH + f"imgs/spec/test/sample.npz")

# Arrange
@pytest.fixture
def c():
    return Frame2D.CHN
