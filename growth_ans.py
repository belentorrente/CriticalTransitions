import numpy.fft
from pcraster.framework import *
from numpy import *
import pylab
from skimage import exposure


class Growth(DynamicModel):

    def __init__(self, map_len, d_rate, out_dir):
        DynamicModel.__init__(self)
        # Init properties used in the model
        self.numberOfNeighbours = None
        self.d_rate = d_rate
        self.out_dir = out_dir
        self.D = None
        self.x = None
        self.K = None
        self.cI = None
        self.c = None
        self.r = None
        self.map_len = map_len  # Required to propagate lengths
        setclone(map_len, map_len, 1, 0, 0)

        # State variables to store timestep-based results
        self.total_var = []
        self.total_skew = []
        self.total_patch_count = []
        self.avg_patch_size = []

    def initial(self):
        # maximum growth rate
        self.r = 0.08

        # grazing rate
        self.c = 0.1

        # increase in grazing rate
        self.cI = scalar(0.00006)

        # dispersion rate
        self.D = scalar(self.d_rate)

        # carrying capacity, causes vertical displacement of the line only it seems
        self.K = scalar(10)

        # state variable
        self.x = spatial(scalar(8.5))

        # Rook neighbourhood
        self.numberOfNeighbours = window4total(spatial(scalar(1)))

        # Create output dir for this model run
        current_directory = os.getcwd()
        runs_directory = os.path.join(current_directory, "runs")
        final_directory = os.path.join(runs_directory, self.out_dir)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        self.out_dir = final_directory

    def dynamic(self):
        growth = self.r * self.x * (1 - self.x / self.K) - self.c * ((self.x * self.x) / ((self.x * self.x) + 1))
        diffusion = self.D * (window4total(self.x) - self.numberOfNeighbours * self.x)
        growth = growth + diffusion
        self.x = self.x + growth

        self.x = max(self.x + normal(1) / 10, 0)

        self.c = self.c + self.cI

        cell_area = int(self.map_len * self.map_len)  # Default is 40 x 40
        iter_mean = maptotal(self.x) / cell_area
        # Variance for current model iteration
        iter_var = maptotal(sqr(self.x - iter_mean)) / cell_area
        self.total_var.append(float(iter_var))  # Store this variance for final results
        # Standard deviation for current model iteration
        iter_std = sqrt(float(iter_var))
        # Skewness for current  model iteration
        iter_skew = maptotal((self.x - iter_mean) ** 3) / (cell_area * iter_std ** 3)
        self.total_skew.append(float(iter_skew))  # Store this variance for final results

        # FFT calculations
        # Use if clause to only output selected timesteps for FFT
        # if self.currentTimeStep() == 1 or self.currentTimeStep() == 2000 or self.currentTimeStep() == 2090 \
        #        or self.currentTimeStep() == 2500:
        # Prepare plot with two subplots
        f, axarr = pylab.subplots(1, 2)
        f.tight_layout()

        # Calculate and then plot the 2D Fourier "as is"
        ft_raster = numpy.fft.fft2(pcraster.pcr2numpy(self.x, 0))
        ft_raster = numpy.fft.fftshift(ft_raster)
        axarr[0].set(xlabel='', ylabel='Original')
        axarr[0].imshow(numpy.abs(ft_raster))

        # Apply contrast stretching to the "raw" FT obtained above
        p1, p99 = numpy.percentile(numpy.abs(ft_raster), (1, 99))
        img_rescale = exposure.rescale_intensity(numpy.abs(ft_raster), in_range=(p1, p99))
        axarr[1].imshow(img_rescale)
        axarr[1].set(xlabel='', ylabel='Contrast stretching')
        out_filepath = os.path.join(self.out_dir, 'fft' + '_' + str(self.nrTimeSteps()) + '_' + str(self.map_len)
                                    + '_' + str(self.currentTimeStep()) + '.png')

        # set the spacing between subplots
        f.subplots_adjust(left=0.15, bottom=0, right=0.90, top=1, wspace=0.5, hspace=0)
        f.suptitle("2D Fourier transform at time step " + str(self.currentTimeStep()), y=0.8)
        f.tight_layout()
        pylab.savefig(out_filepath, bbox_inches='tight')
        pylab.close()  # close figure once saved to file

        # Calculations towards patch count
        report(self.x, "temp.map")  # save copy of state to temp file to manipulate it later
        clump_map = clump(nominal(pcraster.readmap("temp.map")))
        clump_matrix = numpy.asmatrix(pcraster.pcr2numpy(clump_map, -1))
        min_clump_id = clump_matrix.min()
        max_clump_id = clump_matrix.max()
        self.total_patch_count.append(max_clump_id)
        clump_sizes = []
        for i in range(min_clump_id, max_clump_id):
            curr_total = (clump_matrix == i).sum()
            clump_sizes.append(curr_total)
        if not clump_sizes:
            # Case for one big clump
            self.avg_patch_size.append(self.map_len * self.map_len)
        else:
            # General case
            self.avg_patch_size.append(numpy.matrix(clump_sizes).mean())

        # Export results on last iteration
        if self.currentTimeStep() == self.nrTimeSteps():
            pylab.close()  # Close previous figure in case fft figure above is active
            print("End of model run")
            # Plot all four predictors
            pylab.figure(figsize=(20, 4))
            t = range(0, self.nrTimeSteps())
            pylab.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.90, wspace=0.2, hspace=0)

            pylab.subplot(1, 4, 1)
            pylab.plot(t, numpy.array(self.total_var), 'red')
            pylab.title('Variance over time')

            pylab.subplot(1, 4, 2)
            pylab.plot(t, self.total_skew, 'blue')
            pylab.title('Skewness over time')

            pylab.subplot(1, 4, 3)
            pylab.plot(t, self.total_patch_count, 'green')
            pylab.title('Patch count over time')

            pylab.subplot(1, 4, 4)
            pylab.plot(t, self.avg_patch_size, 'purple')
            pylab.title('Average patch size over time')

            out_filepath = os.path.join(self.out_dir, "predictors" + '_' + str(self.nrTimeSteps()) + '_'
                                        + str(self.map_len) + '.png')
            pylab.savefig(out_filepath)
            pylab.close()


nrOfTimeSteps = 20
cellSizes = [40, 80, 160]
dispersionRates = [0.1, 0.01, 0.001]
for cellSize in cellSizes:
    for dispersionRate in dispersionRates:
        myModel = Growth(cellSize, dispersionRate, "run_" + str(nrOfTimeSteps) + "_" + str(cellSize) + "_"
                         + str(dispersionRate))
        dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
        dynamicModel.run()
