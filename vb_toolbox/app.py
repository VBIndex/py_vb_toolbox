#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <l.campos@fz-juelich.de>
#
# Distributed under terms of the GNU license.

import argparse
import multiprocessing
import nibabel
import numpy as np
import textwrap as _textwrap
import vb_toolbox.io as io
import vb_toolbox.vb_index as vb
import sys
import os

class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n ')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = _textwrap.fill(paragraph, width, initial_indent=indent, subsequent_indent=indent) + '\n\n'
            multiline_text = multiline_text + formatted_paragraph
        return multiline_text

def create_parser():
    authors = '''authors:
    |n   Lucas da Costa Campos (lqccampos (at) gmail.com) and Claude J Bajada (claude.bajada (at) um.edu.mt).'''
    copyright = '''copyright:|n
        This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.|n
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.|n
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses>.
    '''

    parser = argparse.ArgumentParser(description='Calculate the Vogt-Bailey index of a dataset. For more information, check https://github.com/VBIndex/py_vb_toolbox.',
                                     epilog=authors + " |n " + copyright,
                                     formatter_class=MultilineFormatter)
    parser.add_argument('-j', '--jobs', metavar='N', type=int, nargs=1,
                        default=[multiprocessing.cpu_count()], help="""Maximum
                        number of jobs to be used. If abscent, one job per CPU
                        will be spawned.""")

    parser.add_argument('-n', '--norm', metavar='norm', type=str, nargs=1,
                        default=["geig"], help="""Laplacian normalization to be
                        used. Possibilities are "geig", "unnorm", "rw" and
                        "sym". Defaults to geig.""")

    parser.add_argument('-fb', '--full-brain', action='store_true',
                        help="""Calculate full brain feature gradient analysis.""")

    parser.add_argument('-hy', '--hybrid', action='store_true',
                        help="""Calculate VB index with hybrid approach.""")

    parser.add_argument('-vm', '--volmask', metavar='file', type=str,
                               nargs=1, default=None, help="""Nifti file containing the whole brain mask
                               in volumetric space. This flag must be set if computing hybrid VB.""")

    parser.add_argument('-m', '--mask', metavar='file', type=str,
                               nargs=1, help="""File containing the labels to
                               identify the cortex, rather than the medial
                               brain structures. This flag must be set for
                               normal analysis and full brain analysis.""")

    parser.add_argument('-c', '--clusters', metavar='file', type=str, nargs=1, default=None,
                        help="""File containing the surface clusters. Cluster
                        with index 0 are expected to denote the medial brain
                        structures and will be ignored.""")
                        
    parser.add_argument('-t', '--tol', metavar='tolerance', type=float, nargs=1,
                        default=["def_tol"], help="""Residual tolerance (stopping criterion) for LOBPCG. 
                        Default value = sqrt(10e-18)*n, where n is the number of nodes per graph.""")
                        
    parser.add_argument('-mi', '--maxiter', metavar='max iterations', type=int, nargs=1, default=[50],
                        help="""Maximum number of iterations for LOBPCG. Defaults to 50.""")

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('-s', '--surface', metavar='file', type=str,
                               nargs=1, help="""File containing the surface
                                              mesh.""", required=True)

    requiredNamed.add_argument('-d', '--data', metavar='file', type=str,
                               nargs=1, help="""File containing the data over
                                              the surface (or volume if hybrid).""", required=True)


    requiredNamed.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Base name for the
                                              output files.""", required=True)

    return parser


def main():

    parser = create_parser()
    args = parser.parse_args()

    n_cpus = args.jobs[0]
    nib_surf, vertices, faces = io.open_gifti_surf(args.surface[0])

    # Get the text contents from the file
    surf_text = open(args.surface[0], 'r', encoding='latin-1')

    hemi = None

    # Check whether the file is left or right cortex
    for line in surf_text:
        if 'CortexLeft' in line:
            hemi = 'CortexLeft'
            break
        elif 'CortexRight' in line:
            hemi = 'CortexRight'
            break

    # Add the cortex information to the beginning of the meta data
    if hemi:
        nib_surf.meta.data.insert(0, nibabel.gifti.GiftiNVPairs('AnatomicalStructurePrimary', hemi))

    nib = nibabel.load(args.data[0])
    if args.hybrid:
        data = np.array(nib.dataobj)
        affine = nib.affine
    else:
        if len(nib.darrays) > 1:
            data = np.array([n.data for n in nib.darrays]).transpose()
        else:
            data = nib.darrays[0].data

    if args.full_brain:
        print("Running full brain analysis")
        if args.mask is None:
            sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
            sys.exit(2)
            quit()
        # Read labels
        _, labels = io.open_gifti(args.mask[0])
        cort_index = np.array(labels, np.bool)
        Z = np.array(cort_index, dtype=np.int)
        try:
            result = vb.vb_cluster(vertices, faces, n_cpus, data, Z, args.norm[0], args.tol[0], args.maxiter[0], args.output[0] + "." + args.norm[0], nib_surf)
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()

    elif args.clusters is None:
        if args.hybrid:
            print("Running searchlight analysis with hybrid approach")
            if args.mask is None:
                sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
                sys.exit(2)
                quit()
            if args.volmask is None:
                sys.stderr.write("A volumetric mask file must be provided through the --volmask flag. See --help")
                sys.exit(2)
                quit()
          
            # Read labels
            _, labels = io.open_gifti(args.mask[0])
            cort_index = np.array(labels, np.bool)
            # Read brain mask
            brainmask = nibabel.load(args.volmask[0])
            brainmask = np.array(brainmask.dataobj)
            try:
                result = vb.vb_hybrid(vertices, brainmask, affine, n_cpus, data, args.norm[0], cort_index, args.tol[0], args.maxiter[0], args.output[0] + "." + args.norm[0], nib_surf)
            except Exception as error:
                sys.stderr.write(str(error))
                sys.exit(2)
                quit()
        else:
            print("Running searchlight analysis")
            if args.mask is None:
                sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
                sys.exit(2)
                quit()
            # Read labels
            _, labels = io.open_gifti(args.mask[0])
            cort_index = np.array(labels, np.bool)
            try:
                result = vb.vb_index(vertices, faces, n_cpus, data, args.norm[0], cort_index, args.tol[0], args.maxiter[0], args.output[0] + "." + args.norm[0], nib_surf)
            except Exception as error:
                sys.stderr.write(str(error))
                sys.exit(2)
                quit()
    else:
        print("Running ROI analysis")
        if args.clusters is None:
            sys.stderr.write("A cluster file must be provided through the --clusters flag. See --help")
            sys.exit(2)
            quit()
        nib, Z = io.open_gifti(args.clusters[0])
        Z = np.array(Z, dtype=np.int)
        try:
            result = vb.vb_cluster(vertices, faces, n_cpus, data, Z, args.norm[0], args.tol[0], args.maxiter[0], args.output[0] + "." + args.norm[0], nib_surf)
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()


if __name__ == "__main__":
    main()
