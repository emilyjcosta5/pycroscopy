# -*- coding: utf-8 -*-
"""
Add description

Created on Tue July 02, 2019

@author: Alvin Tan, Emily Costa

"""

from __future__ import division, print_function, absolute_import, unicode_literals
import h5py
import numpy as np
from pyUSID.processing.process import Process
# Set up parallel compute later to run through supercomputer or cluster
from pyUSID.processing.comp_utils import parallel_compute
from pyUSID.io.hdf_utils import create_results_group, write_main_dataset, write_simple_attrs, create_empty_dataset, \
    write_ind_val_dsets
from pyUSID.io.write_utils import Dimension

# Add other needed imports

# TODO: correct implementation of num_pix


class AdaptiveBayesianInference(Process):
    #what parameters do you need?
    def __init__(self, h5_main, *args,  **kwargs):
        """
        What does this class do?
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        Specify the parameters and what they do and if they are optional
        """
        #
        super(AdaptiveBayesianInference, self).__init__(h5_main, **kwargs)

        #now make sure all parameters were inputted correctly
        # Ex. if frequency_filters is None and noise_threshold is None:
        #    raise ValueError('Need to specify at least some noise thresholding / frequency filter')

        # Now do some setting of the variables
        # Ex. self.frequency_filters = frequency_filters

        # Name the process
        # Ex. self.process_name = 'FFT_Filtering'

        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        self.data = None
        # Add other datasets needs
        # Ex. self.filtered_data = None


    def test(self, pix_ind=None, excit_wfm=None, **kwargs):
        """
        Tests the signal filter on a single pixel (randomly chosen unless manually specified) worth of data.
        Parameters
        ----------
        pix_ind : int, optional. default = random
            Index of the pixel whose data will be used for inference
        excit_wfm : array-like, optional. default = None
            Waveform against which the raw and filtered signals will be plotted. This waveform can be a fraction of the
            length of a single pixel's data. For example, in the case of G-mode, where a single scan line is yet to be
            broken down into pixels, the excitation waveform for a single pixel can br provided to automatically
            break the raw and filtered responses also into chunks of the same size.
        Returns
        -------
        fig, axes
        """
        if self.mpi_rank > 0:
            return
        if pix_ind is None:
            pix_ind = np.random.randint(0, high=self.h5_main.shape[0])

        # Return from test function you built seperately (see gmode_utils.test_filter for example)
        return test_filter(self.h5_main[pix_ind], frequency_filters=self.frequency_filters, excit_wfm=excit_wfm,
                           noise_threshold=self.noise_threshold, plot_title='Pos #' + str(pix_ind), show_plots=True,
                           **kwargs)

    def _create_results_datasets(self):
        """
        Creates all the datasets necessary for holding all parameters + data.
        """

        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

        self.parms_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_AdaptiveBayesianInference'})

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        assert isinstance(self.h5_results_grp, h5py.Group)

        self.h5_main.file.flush()

    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets that hold the results
        """
        # Do not worry to much about this now 

    def _write_results_chunk(self):
        """
        Writes data chunks back to the file
        """
        # Get access to the private variable:
        pos_in_batch = self._get_pixels_in_current_batch()

        # Ex. if self.write_filtered:
        #    self.h5_filtered[pos_in_batch, :] = self.filtered_data

        # Process class handles checkpointing.

    def _unit_computation(self, *args, **kwargs):
        """
        Processing per chunk of the dataset
        Parameters
        ----------
        args : list
            Not used
        kwargs : dictionary
            Not used
        """
        # This is where you add the Matlab code you translated. You can add stuff to other Python files in processing, 
        #like gmode_utils or comp_utils, depending on what your function does. Just add core code here.
