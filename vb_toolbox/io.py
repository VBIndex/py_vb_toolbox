#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the GNU license.

import nibabel
import numpy as np

def open_gifti_surf(filename):
    """
    Helper function to read the surface from a gifti file.

    Parameters
    ----------
    filename: string
    String containing the name of the file to be opened.

    Returns
    -------
    nib: Nibabel object
    vertices: (N, 3) numpy vector
              Vertices of the mesh
    faces: (M, 3) numoy vector
            Faces of the mesh.
    """
    nib = nibabel.load(filename)
    return nib, nib.get_arrays_from_intent('pointset')[0].data,  nib.get_arrays_from_intent('triangle')[0].data

def open_gifti(filename):
    """
    Helper function to read data from a gifti file.

    Parameters
    ----------
    filename: string
    String containing the name of the file to be opened.

    Returns
    -------
    nib: Nibabel object
    data: (M, N) numpy array
          Data in the file.
    """
    nib = nibabel.load(filename)

    # We use the first data as this is agnostic to the intent. In the future we
    # might want to change it.
    return nib, nib.darrays[0].data

def save_gifti(og_img, data, filename):
    """
    Helper function to write data into a gifti file.

    Parameters
    ----------
    og_img: Nibabel object
    data: (M, N) numpy array
          Data to write into the file. M must math the number of vertices in the mesh
    filename: string
              String containing the name of the file to be saved
    """
    # For some reason, wc_view demands float32
    data_array = nibabel.gifti.gifti.GiftiDataArray(np.array(data, dtype=np.float32))

    # Create a meta object containing the cortex information
    new_meta = nibabel.gifti.gifti.GiftiMetaData(og_img.meta.data[0])
    new_nib = nibabel.gifti.gifti.GiftiImage(darrays=[data_array], meta=new_meta)

    nibabel.save(new_nib, filename)
