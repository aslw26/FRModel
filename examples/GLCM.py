from frmodel.base.D2 import Frame2D

f = Frame2D.from_image("sample.jpg", scale=0.2)
g = f.get_chns(self_=True, glcm=f.GLCM(radius=3, bins=8))
fpl = g.plot()
fpl.image().savefig("sample_out.jpg")