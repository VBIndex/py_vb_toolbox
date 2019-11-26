#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <l.campos@fz-juelich.de>
#
# Distributed under terms of the GPL license.

import nibabel
import numpy as np

def open_gifti_surf(filename):
    """Helper function to read the surface and data from a gifti file"""
    nib = nibabel.load(filename)
    return nib, nib.darrays[0].data,  nib.darrays[1].data

def open_gifti(filename):
    """Helper function to read the data from a gifti file"""
    nib = nibabel.load(filename)
    return nib, nib.darrays[0].data

def save_gifti(og_img, data, filename):
    """Helper function to save data into a new gifti file"""
    # For some reason, wc_view demands float32
    data_array = nibabel.gifti.gifti.GiftiDataArray(np.array(data, dtype=np.float32))
    new_nib = nibabel.gifti.gifti.GiftiImage(darrays=[data_array])
    nibabel.save(new_nib, filename)
