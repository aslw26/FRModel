from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, List, TYPE_CHECKING

import numpy as np

from frmodel.base import CONSTS
from frmodel.base.D2.frame._cy_fast_glcm2 import CyGLCM

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

class _Frame2DChannelFastGLCM(ABC):
    """ This re-implements Wang Jifei's Fast GLCM Script by adding the option of binarization. """

    @dataclass
    class GLCM:
        """ This holds all GLCM parameters to pass into get_glcm

        Note that contrast, correlation, asm takes arguments similarly to how get_chns work.

        Entropy has been replaced by ASM. Angular Second Moment

        e.g. contrast=[f.CHN.HSV]
        """
        by:     int = 1
        radius: int = 2
        bins:   int = 8

        channels: List[CONSTS.CHN] = field(default_factory=lambda: [])

    def get_glcm(self: 'Frame2D', glcm:GLCM) -> Tuple[np.ndarray, List[str]]:
        """ This will get the GLCM statistics for this window

        Details on how GLCM works is shown on the wiki.

        :param glcm: A GLCM Class, this class holds all parameters.
        """

        # FAST GLCM
        channels = glcm.channels if glcm.channels else self.labels.keys()
        data = CyGLCM(self[..., channels].data.astype(np.float32), glcm.radius, glcm.bins)\
            .create_glcm()
        data = data.swapaxes(-2,-1).reshape([*data.shape[:2], -1])

        labels = []

        labels.extend(CONSTS.CHN.GLCM.CON( list(self._util_flatten(channels))))
        labels.extend(CONSTS.CHN.GLCM.COR( list(self._util_flatten(channels))))
        labels.extend(CONSTS.CHN.GLCM.ASM( list(self._util_flatten(channels))))
        labels.extend(CONSTS.CHN.GLCM.MEAN(list(self._util_flatten(channels))))
        labels.extend(CONSTS.CHN.GLCM.VAR( list(self._util_flatten(channels))))

        return data, labels
