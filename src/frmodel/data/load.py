import os.path
from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd

from frmodel.base.D2 import Frame2D


@dataclass
class Tree:
    name: str
    frame: Frame2D

def load_spec(dir_path: str, scale: float, bounds_path: str = "bounds.csv") -> Tuple[Frame2D, List[Tree]]:
    """ Quick loads a spec dataset. The required files must be present.

    Required:
    result_Red.tif, Green, Blue, RedEdge, NIR

    Optional:
    bounds.csv

    """

    assert os.path.isfile(dir_path + "result_Red.tif") \
       and os.path.isfile(dir_path + "result_Green.tif") \
       and os.path.isfile(dir_path + "result_Blue.tif") \
       and os.path.isfile(dir_path + "result_RedEdge.tif") \
       and os.path.isfile(dir_path + "result_NIR.tif"),\
        f"Some required files are missing, make sure that {dir_path}result_xxx.tif exists"

    f = Frame2D.from_image_spec(
        dir_path + "result_Red.tif",
        dir_path + "result_Green.tif",
        dir_path + "result_Blue.tif",
        dir_path + "result_RedEdge.tif",
        dir_path + "result_NIR.tif",
        scale=scale
    )

    trees = []
    if os.path.isfile(dir_path + bounds_path):
        bounds = pd.read_csv(dir_path + bounds_path, "|", header=None)

        for _, r in bounds.iterrows():
            r_ = (r[1:] * scale).astype(int)
            tree = f[r_[1]:r_[2],r_[3]:r_[4]]
            trees.append(Tree(str(r[0]), tree))

    return f, trees
