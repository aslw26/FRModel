import numpy as np
cimport numpy as np
cimport cython
from skimage.util import view_as_windows

ctypedef np.uint8_t DTYPE_t8
ctypedef np.uint16_t DTYPE_t16
ctypedef np.uint32_t DTYPE_t32
ctypedef np.float32_t DTYPE_ft32
from cython.parallel cimport prange
from tqdm import tqdm
from libc.math cimport sqrt


cdef class GLCM:
    cdef public DTYPE_t8 radius, bins, diameter

    def __init__(self, DTYPE_t8 radius, DTYPE_t8 bins):
        self.radius = radius
        self.diameter = radius * 2 + 1
        self.bins = bins
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cy_glcm(self, np.ndarray[DTYPE_t32, ndim=3] ar):
        """

        :param ar: R C CH
        :return:
        """

        cdef np.ndarray ar_bin = self._binarize(ar)
        cdef DTYPE_t8 chs = ar_bin.shape[2]
        cdef np.ndarray glcm = self._empty_glcm(ar)
        for ch in tqdm(range(chs)):
            pairs = self._pair(ar_bin[..., ch])
            for pair in pairs:
                # Pair: Tuple
                self._populate_glcm(pair[0], pair[1], glcm[...,ch])

        return glcm + glcm.swapaxes(0,1)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _binarize(self, np.ndarray[DTYPE_t32, ndim=3] ar) -> np.ndarray:
        """ This binarizes the 2D image by its min-max """
        return (((ar - ar.min()) / ar.max()) * (self.bins - 1)).astype(np.uint8)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _empty_glcm(self, np.ndarray[DTYPE_t32, ndim=3] ar) -> np.ndarray:
        """ Creates an empty GLCM

        Shape is BIN, BIN, WRS, WCS, CH
        """

        # TODO: Make this only for every channel
        return np.zeros(shape=[self.bins, self.bins,
                               ar.shape[0] - self.diameter,
                               ar.shape[1] - self.diameter,
                               ar.shape[2]],
                        dtype=np.uint8)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _empty_feature(self, np.ndarray[DTYPE_t32, ndim=3] ar) -> np.ndarray:
        """ Creates an empty GLCM

        Shape is BIN, BIN, WRS, WCS, CH
        """
        return np.zeros(shape=[self.bins, self.bins,
                               ar.shape[0] - self.diameter,
                               ar.shape[1] - self.diameter,
                               ar.shape[2]],
                        dtype=np.uint8)


    @cython.boundscheck(False)
    def _pair(self, np.ndarray[DTYPE_t8, ndim=2] ar):
        ar_w = view_as_windows(ar, (self.diameter, self.diameter))
        pair_h = (ar_w[:-1, :-1], ar_w[:-1, 1:])
        pair_v = (ar_w[:-1, :], ar_w[1:, :])
        pair_se = (ar_w[:-1, :-1], ar_w[1:, 1:])
        pair_ne = (ar_w[1:, :-1], ar_w[1:, 1:])
        return pair_h, pair_v, pair_se, pair_ne


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _populate_glcm(self,
                       np.ndarray[DTYPE_t8, ndim=4] pair_i,
                       np.ndarray[DTYPE_t8, ndim=4] pair_j,
                       np.ndarray[DTYPE_t8, ndim=4] glcm):
        """ The ar would be WR, WC, CR, CC

        :param pair_i: WR WC CR CC
        :param pair_j: WR WC CR CC
        :param glcm: BIN BIN WR WC
        :return:
        """
        cdef DTYPE_t8 wrs = glcm.shape[2]
        cdef DTYPE_t8 wcs = glcm.shape[3]
        for wr in range(wrs):
            for wc in range(wcs):
                self._populate_glcm_single(pair_i[wr, wc], pair_j[wr, wc], glcm[..., wr, wc])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _populate_glcm_single(self,
                              np.ndarray[DTYPE_t8, ndim=2] pair_i,
                              np.ndarray[DTYPE_t8, ndim=2] pair_j,
                              np.ndarray[DTYPE_t8, ndim=2] glcm):
        """

        :param pair_i: CR CC
        :param pair_j: CR CC
        :param glcm: BIN BIN
        :return:
        """
        cdef DTYPE_t8 crs = pair_i.shape[0]
        cdef DTYPE_t8 ccs = pair_i.shape[1]
        cdef DTYPE_t8 i = 0
        cdef DTYPE_t8 j = 0
        for cr in range(crs):
            for cc in range(ccs):
                i = pair_i[cr, cc]
                j = pair_j[cr, cc]
                glcm[i, j] += 1

                # Contrast += ((i - j) ** 2) / n
                # ASM += sum of all squared / n**2


    #
    # # Used to call these windows_a and windows_b, but it's easier if it follows the
    # # tutorial's naming convention.
    # cdef DTYPE_t8 [:, :, :, :, :] i_v = windows_i
    # cdef DTYPE_t8 [:, :, :, :, :] j_v = windows_j
    #
    # cdef unsigned short window_rows = <unsigned int> windows_i.shape[0]
    # cdef unsigned short window_cols = <unsigned int> windows_i.shape[1]
    # cdef short wi_r = 0, wi_c = 0
    #
    # cdef unsigned int cell_rows = <unsigned int> windows_i.shape[2]
    # cdef unsigned int cell_cols = <unsigned int> windows_i.shape[3]
    # cdef short c_r = 0, c_c = 0
    #
    # cdef unsigned char channels = <char>windows_i.shape[4]
    # cdef char ch = 0
    # cdef short i = 0
    # cdef short j = 0
    #
    # # Vals are just temporary variables so as to facilitate clean code.
    # cdef float glcm_val   = 0.0
    # cdef float mean_i_val = 0.0
    # cdef float mean_j_val = 0.0
    # cdef float var_i_val  = 0.0
    # cdef float var_j_val  = 0.0
    #
    # # Finds the maximum out of the 2 windows
    # # We do this instead of a user input because it's cheap and it reduces the required amount of
    # # windows to the minimal.
    # cdef char glcm_max_value = np.max([windows_i, windows_j]) + 1
    #
    # glcm        = np.zeros([glcm_max_value, glcm_max_value,
    #                         window_rows, window_cols, channels], dtype=np.uint8)
    # contrast    = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # correlation = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # asm         = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # mean        = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # mean_i      = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # mean_j      = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # var         = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # var_i       = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    # var_j       = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    #
    # # We declare all of the required Views here.
    # # All views are prepended with _v.
    #
    # # To Clarify, GLCM uses these as Dimensions
    # # i, j, wi_r, wi_c, ch
    # cdef unsigned char  [:, :, :, :, :] glcm_v        = glcm
    # cdef float          [:, :, :]       contrast_v    = contrast
    # cdef float          [:, :, :]       correlation_v = correlation
    # cdef float          [:, :, :]       asm_v         = asm
    # cdef float          [:, :, :]       mean_v        = mean
    # cdef float          [:, :, :]       mean_i_v      = mean_i
    # cdef float          [:, :, :]       mean_j_v      = mean_j
    # cdef float          [:, :, :]       var_v         = var
    # cdef float          [:, :, :]       var_i_v       = var_i
    # cdef float          [:, :, :]       var_j_v       = var_j
    #
    # # ------------------------
    # # GLCM CONSTRUCTION
    # # ------------------------
    # # This is the part where GLCM is for loop generated.
    # # It's very quick actually.
    #
    # # The outer loop is required for verbose tqdm
    # # prange will not work if there are any Python objects within
    # # So the options are exclusive, but verbose is more important
    # for wi_r in tqdm(range(window_rows), disable=not verbose,
    #                  desc="GLCM Construction Pass"):
    #     for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
    #         for ch in prange(channels, schedule='dynamic'):
    #             for c_r in prange(cell_rows, schedule='dynamic'):
    #                 for c_c in prange(cell_cols, schedule='dynamic'):
    #                     i = <short>i_v[wi_r, wi_c, c_r, c_c, ch]
    #                     j = <short>j_v[wi_r, wi_c, c_r, c_c, ch]
    #                     glcm_v[i, j, wi_r, wi_c, ch] += 1
    #                     glcm_v[j, i, wi_r, wi_c, ch] += 1
    #
    # # ------------------------
    # # CONTRAST, ASM, MEAN
    # # ------------------------
    # # Contrast and ASM is generated here.
    # # Correlation takes 3 passes, for the detailed explanation, read the journal.
    # # In simple words, Corr needs the variance, variance needs the mean, so it requires
    # # multiple passes for Corr to be done.
    #
    # for wi_r in tqdm(range(window_rows), disable=not verbose,
    #                  desc="GLCM Contrast, ASM, Mean Pass"):
    #     for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
    #         for ch in prange(channels, schedule='dynamic'):
    #             for i in prange(glcm_max_value, schedule='dynamic'):
    #                 for j in prange(glcm_max_value, schedule='dynamic'):
    #                     glcm_val = glcm_v[i, j, wi_r, wi_c, ch] / glcm_max_value
    #                     if glcm_val != 0:
    #                         contrast_v [wi_r, wi_c, ch] += glcm_val * ((i - j) ** 2)
    #                         asm_v      [wi_r, wi_c, ch] += glcm_val ** 2
    #                         mean_i_v   [wi_r, wi_c, ch] += glcm_val * i
    #                         mean_j_v   [wi_r, wi_c, ch] += glcm_val * j
    #
    # # ------------------------
    # # VARIANCE
    # # ------------------------
    # # Only VARIANCE is done here.
    #
    # for wi_r in tqdm(range(window_rows), disable=not verbose,
    #                  desc="GLCM Variance Pass"):
    #     for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
    #         for ch in prange(channels, schedule='dynamic'):
    #             for i in prange(glcm_max_value, schedule='dynamic'):
    #                 for j in prange(glcm_max_value, schedule='dynamic'):
    #                     glcm_val = glcm_v[i, j, wi_r, wi_c, ch] / glcm_max_value
    #                     mean_i_val = mean_i_v[wi_r, wi_c, ch]
    #                     mean_j_val = mean_j_v[wi_r, wi_c, ch]
    #                     if glcm_val != 0:
    #                         var_i_v[wi_r, wi_c, ch] += glcm_val * (i - mean_i_val) ** 2
    #                         var_j_v[wi_r, wi_c, ch] += glcm_val * (j - mean_j_val) ** 2
    #
    # # ---------------------------
    # # CORRELATION, MEAN, VARIANCE
    # # ---------------------------
    # # Mean and Variance are also features we want, so we just merge them by averaging
    #
    # for wi_r in tqdm(range(window_rows), disable=not verbose,
    #                  desc="GLCM Correlation, Mean, Variance Merge Pass"):
    #     for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
    #         for ch in prange(channels,              schedule='dynamic'):
    #             for i in prange(glcm_max_value,     schedule='dynamic'):
    #                 for j in prange(glcm_max_value, schedule='dynamic'):
    #                     glcm_val = glcm_v[i, j, wi_r, wi_c, ch] / glcm_max_value
    #                     mean_i_val = mean_i_v[wi_r, wi_c, ch]
    #                     mean_j_val = mean_j_v[wi_r, wi_c, ch]
    #                     var_i_val  = var_i_v[wi_r, wi_c, ch]
    #                     var_j_val  = var_j_v[wi_r, wi_c, ch]
    #                     if glcm_val != 0:
    #                         if var_i_val != 0 and var_j_val != 0:
    #                             correlation_v[wi_r, wi_c, ch] += glcm_val * (
    #                                 (i - mean_i_val) * (j - mean_j_val) /
    #                                 sqrt(var_i_val * var_j_val)
    #                             )
    #                     mean_v[wi_r, wi_c, ch] = (mean_i_val + mean_j_val) / 2
    #                     var_v[wi_r, wi_c, ch]  = (var_i_val + var_j_val) / 2



