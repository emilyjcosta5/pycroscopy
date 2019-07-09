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


class AdaptiveBayesianInference(Process):
    def __init__(self, h5_main, **kwargs):
        """
        Bayesian inference is done on h5py dataset object that has already been filtered
        and reshaped.
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        kwargs : (Optional) dictionary
            Please see Process class for additional inputs
        """
       super(AdaptiveBayesianInference, self).__init__(h5_main, **kwargs)

        #now make sure all parameters were inputted correctly
        # Ex. if frequency_filters is None and noise_threshold is None:
        #    raise ValueError('Need to specify at least some noise thresholding / frequency filter')
        # This determines how to parse down the data (i.e. used_data = actual_data[::parse_mod]).
        # If parse_mod == 1, then we use the entire dataset. We may be able to input this as an
        # argument, but for now this is just to improve maintainability.
        self.parse_mod = 4
	if self.verbose: print("parsed data")
        # Now do some setting of the variables
        # Ex. self.frequency_filters = frequency_filters
        self.full_V = np.array([float(v) for v in self.h5_main.h5_spec_vals[()][0]][::self.parse_mod])
	if self.verbose: print("V set up")
        # Name the process
        # Ex. self.process_name = 'FFT_Filtering'
        self.process_name = 'Adaptive_Bayesian'

        # Honestly no idea what this line does
        # This line checks to make sure this data has not already been processed
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        self.data = None
        # Add other datasets needs
        # Ex. self.filtered_data = None

        # A couple constants and vectors we will be using
        self.full_V, self.shift_index, self.split_index = get_shift_and_split_indices(self.full_V)
        self.M, self.dx, self.x = get_M_dx_x(V0=max(full_V), M=25)
	if self.verbose: print("data and variables set up")
        # These will be the results from the processed chunks
        self.R = None
        self.R_sig = None
        self.capacitance = None
        self.i_recon = None
        self.i_corrected = None
        self.shifted_i_meas = None
	
        # These are the actual databases
        self.h5_R = None
        self.h5_R_sig = None
        self.h5_capacitance = None
        self.h5_i_recon = None
        self.h5_i_corrected = None
	if self.verbose: ("empty results set up")

        # Make full_V and num_pixels attributes
        self.params_dict = dict()
        self.params_dict["num_pixels"] = self.h5_main[()].shape[0]
        self.params_dict["full_V"] = self.full_V
        self.params_dict["shift_index"] = self.shift_index
        self.params_dict["split_index"] = self.split_index
        self.params_dict["M"] = self.M
        self.params_dict["dx"] = self.dx
        self.params_dict["x"] = self.x
	if self.verbose: ("attributes set up")
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
        return process_pixel(full_i_meas, self.full_V, self.split_index, self.M, self.dx, self.x, graph=True, verbose=True)

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
        # Note, each pixel of the datasets will hold the forward and reverse sweeps concatenated together.
        # R and R_sig are plotted against [x, -x], and i_recon and i_corrected are plotted against full_V.
        self.h5_R = h5_results_grp.create_dataset("R", shape=(self.h5_main.shape[0], 2*self.M), dtype=np.float)
        self.h5_R_sig = h5_results_grp.create_dataset("R_sig", shape=(self.h5_main.shape[0], 2*self.M), dtype=np.float)
        self.h5_capacitance = h5_results_grp.create_dataset("capacitance", shape=(self.h5_main.shape[0], 2), dtype=np.float)
        self.h5_i_recon = h5_results_grp.create_dataset("i_recon", shape=(self.h5_main.shape[0], self.full_V.size), dtype=np.float)
        self.h5_i_corrected = h5_results_grp.create_dataset("i_corrected", shape=(self.h5_main.shape[0], self.full_V.size), dtype=np.float)
	if self.verbose: print("results datasets set up")
        self.h5_main.file.flush()

    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets that hold the results
        """
        # Do not worry to much about this now 
        self.h5_R = self.h5_results_grp["R"]
        self.h5_R_sig = self.h5_results_grp["R_sig"]
        self.h5_capacitance = self.h5_results_grp["capacitance"]
        self.h5_i_recon = self.h5_results_grp["i_recon"]
        self.h5_i_corrected = self.h5_results_grp["i_corrected"]

    def _write_results_chunk(self):
        """
        Writes data chunks back to the file
        """
        # Get access to the private variable:
        pos_in_batch = self._get_pixels_in_current_batch()

        # Ex. if self.write_filtered:
        #    self.h5_filtered[pos_in_batch, :] = self.filtered_data
        self.h5_R[pos_in_batch, :] = self.R
        self.h5_R_sig[pos_in_batch, :] = self.R_sig
        self.h5_capacitance[pos_in_batch, :] = self.capacitance
        self.h5_i_recon[pos_in_batch, :] = self.i_recon
        self.h5_i_corrected[pos_in_batch, :] = self.i_corrected
	if self.verbose: print("results written back to file")
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
        self.shifted_i_meas = parallel_compute(self.data[:, ::self.parse_mod], get_shifted_response, cores=self._cores, func_args=[self.shift_index])
        all_data = parallel_compute(self.shifted_i_meas, process_pixel, cores=self._cores, func_args=[self.full_V, self.split_index, self.M, self.dx, self.x])

        # Since process_pixel returns a tuple, parse the list of tuples into individual lists
        # Note, results are (R, R_sig, capacitance, i_recon, i_corrected)
        # and R, R_sig, i_recon, and i_corrected are column vectors, which are funky to work with
        self.R = np.array([result[0].T[0] for result in all_data]).astype(np.float)
        self.R_sig = np.array([result[1].T[0] for result in all_data]).astype(np.float)
        self.capacitance = np.array([result[2] for result in all_data])
        self.i_recon = np.array([result[3].T[0] for result in all_data]).astype(np.float)
        self.i_corrected = np.array([result[4].T[0] for result in all_data]).astype(np.float)
	if self.verbose: print("chunk computed")


























