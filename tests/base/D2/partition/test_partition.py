from tests.base.D2Fixture.test_fixture import f, c

def test_view_windows(f, c):
    height = f.height()
    width = f.width()
    step = 100
    windows = f.view_windows(height // 2, width // 2, step, step)
    assert ((height - (height // 2)) // step + 1,  # +1: Fencepost error
             (width - (width // 2)) // step + 1,   # +1: Fencepost error
             height // 2, width // 2, f.shape[-1]) == windows.shape

def test_view_blocks(f, c):
    height = f.height()
    width = f.width()
    blocks = 5
    windows = f.view_blocks(height // blocks, width // blocks)
    assert (blocks, blocks, height // blocks, width // blocks, f.shape[-1]) == windows.shape
