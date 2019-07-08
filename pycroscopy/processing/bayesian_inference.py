# -*- coding: utf-8 -*-
"""
Runs a full Bayesian inference algorithm on spectroscopic data for ultrafast current imaging (see paper below)
https://www.nature.com/articles/s41467-017-02455-7

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

import time
import math
import scipy.linalg as spla 
import pycroscopy as px 
import pyUSID as usid

from .bayesian_utils import get_shift_and_split_indices
from .bayesian_utils import process_pixel


# TODO: correct implementation of num_pix


class AdaptiveBayesianInference(Process):
    # Note: this assumes use of reshaped data
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

        # This determines how to parse down the data (i.e. used_data = actual_data[::parse_mod]).
        # If parse_mod == 1, then we use the entire dataset. We may be able to input this as an
        # argument, but for now this is just to improve maintainability.
        self.parse_mod = 4

        # Now do some setting of the variables
        # Ex. self.frequency_filters = frequency_filters
        self.full_V = np.array([float(v) for v in self.h5_main.h5_spec_vals[()][0]][::self.parse_mod])

        # Name the process
        # Ex. self.process_name = 'FFT_Filtering'
        self.process_name = 'Adaptive_Bayesian'

        # Honestly no idea what this line does
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        self.data = None
        # Add other datasets needs
        # Ex. self.filtered_data = None

        # A couple constants and vectors we will be using
        self.full_V, self.shift_index, self.split_index = get_shift_and_split_indices(self.full_V)
        self.M, self.dx, self.x = get_M_dx_x(V0=max(full_V), M=25)

        # These will be the results from the processed chunks
        self.R = None
        self.R_sig = None
        self.i_recon = None
        self.i_corrected = None

        # These are the actual databases
        self.h5_R = None
        self.h5_R_sig = None
        self.h5_i_recon = None
        self.h5_i_corrected = None

        # Make full_V and num_pixels attributes
        self.params_dict = dict()
        self.params_dict["num_pixels"] = self.h5_main[()].shape[0]
        self.params_dict["full_V"] = self.full_V
        self.params_dict["shift_index"] = self.shift_index
        self.params_dict["split_index"] = self.split_index
        self.params_dict["M"] = self.M
        self.params_dict["dx"] = self.dx
        self.params_dict["x"] = self.x

    def test(self, pix_ind=None):
        """
        Tests the Bayesian inference on a single pixel (randomly chosen unless manually specified) worth of data.
        Displays the resulting figure (resistances with variance, and reconstructed current)
            before returning the same figure.
        Parameters
        ----------
        pix_ind : int, optional. default = random
            Index of the pixel whose data will be used for inference
        Returns
        -------
        fig, axes
        """
        if self.mpi_rank > 0:
            return
        if pix_ind is None:
            pix_ind = np.random.randint(0, high=self.h5_main.shape[0])

        full_i_meas = get_shifted_response(self.h5_resh[pix_ind, ::self.parse_mod], self.shift_index)

        # Return from test function you built seperately (see gmode_utils.test_filter for example)
        return process_pixel(self.full_V, full_i_meas, self.split_index, self.M, self.dx, self.x, graph=True, verbose=True)

    def _create_results_datasets(self):
        """
        Creates all the datasets necessary for holding all parameters + data.
        """

        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

        self.parms_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_AdaptiveBayesianInference'})

        # Write in our full_V and num_pixels as attributes to this new group
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        assert isinstance(self.h5_results_grp, h5py.Group)

        # If we ended up parsing down the data, create new spectral datasets (i.e. smaller full_V's)
        if self.parse_mod != 1:
            h5_spec_inds_new, h5_spec_vals_new = write_ind_val_dsets(self.h5_results_grp, self.full_V.shape, is_spectral=True)
            h5_spec_vals_new[()] = self.full_V
        else:
            h5_spec_inds_new = self.h5_main.h5_spec_inds
            h5_spec_vals_new = self.h5_main.h5_spec_vals

        # Initialize our datasets
        self.h5_R = h5_results_grp.create_dataset("R", shape=(self.h5_main.shape[0], 1024), dtype=np.int)
        self.h5_R_sig = None
        self.h5_i_recon = None
        self.h5_i_corrected = None

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


























