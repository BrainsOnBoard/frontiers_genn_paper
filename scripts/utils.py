# Import modules
import csv
import numpy as np
import os
import seaborn as sns
import sys
import tempfile

# Import classes
from matplotlib.ticker import ScalarFormatter

def remove_axis_junk(axis):
    # Turn off grid
    axis.xaxis.grid(False)
    axis.yaxis.grid(False)

# OMFG matplotlib, REALLY! How is this not normal behaviour
class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag


def save_raster_figure(figure, filename):
    # Create a temporary png file
    with tempfile.NamedTemporaryFile(suffix=".png") as temp:
        # Create high-resolution png file
        figure.savefig(temp.name, dpi=1200)

        # Create high-resolution TIF for publication
        os.system("convert %s -compress lzw %s" % (temp.name, filename + ".tif"))

    # Create low-res PNG for latex
    figure.savefig(filename + ".png", dpi=200)