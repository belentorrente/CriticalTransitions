import numpy.fft
from pcraster.framework import *
from numpy import *
import pylab


class Growth(DynamicModel):

    def __init__(self, map_len):
        DynamicModel.__init__(self)
        # Init properties used in the model
        self.numberOfNeighbours = None
        self.D = None
        self.x = None
        self.K = None
        self.cI = None
        self.c = None
        self.r = None
        self.map_len = map_len  # Required to propagate lengths
        # setclone('clone.map') # This default map uses a 40x40 matrix
        # Override default matrix size
        setclone(map_len, map_len, 1, 0, 0)

        # State variables to store timestep-based results
        self.total_var = []
        self.total_skew = []

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

        a_uniform_map = uniform(1)
        self.report(a_uniform_map, 'uni')

    def dynamic(self):
        growth = self.r * self.x * (1 - self.x / self.K) - self.c * ((self.x * self.x) / ((self.x * self.x) + 1))
        diffusion = self.D * (window4total(self.x) - self.numberOfNeighbours * self.x)
        growth = growth + diffusion

        self.x = self.x + growth

        self.x = max(self.x + normal(1) / 10, 0)
        self.report(self.x, 'x')

        self.c = self.c + self.cI

        cell_area = int(self.map_len * self.map_len)  # Default is 40 x 40
        print('Cell area: ', cell_area)
        iter_mean = maptotal(self.x) / cell_area
        # Variance for current model iteration
        iter_var = maptotal(sqr(self.x - iter_mean)) / cell_area
        self.total_var.append(float(iter_var))  # Store this variance for final results
        # Standard deviation for current model iteration
        iter_std = sqrt(float(iter_var))
        # Skewness for current  model iteration
        iter_skew = maptotal((self.x - iter_mean) ** 3) / cell_area
        self.total_skew.append(float(iter_skew))  # Store this variance for final results

        self.report(iter_var, 'var')
        self.report(iter_skew, 'skew')
        # Show results on screen
        print("iter=%d, mean=%f, variance=%f, std=%f, skewness=%f" % (self.currentTimeStep(), float(iter_mean), float(iter_var), iter_std, float(iter_skew)))
        # aguila(self.x)  # Plot this iteration of the model map
        # pylab.imshow(pcraster.pcr2numpy(self.x, 0))  # Uncomment to plot this iteration of model map (same as above)
        # pylab.show()  # Uncomment to plot this iteration of model map (same as above)

        ft_raster = numpy.fft.fft2(pcraster.pcr2numpy(self.x, 0))
        ft_raster = numpy.fft.fftshift(ft_raster)
        pylab.imshow(numpy.abs(ft_raster)) # Uncomment to plot fft2
        # pylab.show() # Uncomment to plot fft2
        pylab.savefig('fft' + str(self.currentTimeStep()) + '.png') # Save every fft run
        # input("Press enter to continue") # Uncomment to plot one by one

        # Export results on last iteration
        if self.currentTimeStep() == self.nrTimeSteps():
            print("End of model run")
            print("Variance over time:")
            print(self.total_var)
            print("Skewness over time:")
            print(self.total_var)


nrOfTimeSteps = 7000
cellSize = 40
myModel = Growth(cellSize)
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
dynamicModel.run()
