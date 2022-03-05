from pcraster.framework import *


class Growth(DynamicModel):
    def __init__(self):
        DynamicModel.__init__(self)
        self.numberOfNeighbours = None
        self.D = None
        self.x = None
        self.K = None
        self.cI = None
        self.c = None
        self.r = None
        setclone('clone.map')

    def initial(self):
        # maximum growth rate
        self.r = 0.08

        # grazing rate
        self.c = 0.1

        # increase in grazing rate
        self.cI = scalar(0.00006)

        # carrying capacity, causes vertical displacement of the line only it seems
        self.K = scalar(10)

        # state variable
        self.x = spatial(scalar(8.5))

        # dispersion rate
        self.D = scalar(0.01)

        self.numberOfNeighbours = window4total(spatial(scalar(1)))

    def dynamic(self):
        growth = self.r * self.x * (1 - self.x / self.K) - self.c * ((self.x * self.x) / ((self.x * self.x) + 1))
        diffusion = self.D * (window4total(self.x) - self.numberOfNeighbours * self.x)
        growth = growth + diffusion
        self.x = self.x + growth

        self.x = max(self.x + normal(1) / 10, 0)
        self.report(self.x, 'x')

        self.c = self.c + self.cI

        mean = maptotal(self.x) / (40 * 40)
        var = maptotal(sqr(self.x - mean)) / (40 * 40)
        self.report(var, 'var')


nrOfTimeSteps = 2500
myModel = Growth()
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
dynamicModel.run()
