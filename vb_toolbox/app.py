#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:48:15 2024

@author: bobducks
"""

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 VB Index Team
#
# Distributed under terms of the GNU license.

import argparse
import multiprocessing
import nibabel
import numpy as np
import textwrap as _textwrap
import sys
import vb_index

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
    |n   The VB Index Team (See Contributors Section in the main README)'''
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
    references = '''references:|n
        Bajada, C. J., Campos, L. Q. C., Caspers, S., Muscat, R., Parker, G. J., Ralph, M. A. L., ... & Trujillo-Barreto, N. J. (2020). A tutorial and tool for exploring feature similarity gradients with MRI data. NeuroImage, 221, 117140.|n
        Ciantar, K. G., Farrugia, C., Scerri, K., Xu, T., & Bajada, C. J. (2020). Geometric effects of volume-to-surface mapping of fMRI data. bioRxiv.
    '''
    
    parser = argparse.ArgumentParser(description='Calculate the Vogt-Bailey index of a dataset. For more information, refer to https://github.com/VBIndex/py_vb_toolbox.',
                                     epilog=authors + " |n " + references + " |n " + copyright,
                                     formatter_class=MultilineFormatter)


    parser.add_argument('-e', '--eigenvalue-index', metavar='INDEX', type=int, nargs=1, default=1,
                        help="""Index of the eigenvalue to use. Default is the second smallest eigenvalue (index 1).""")

    parser.add_argument('-j', '--jobs', metavar='N', type=int, nargs=1,
                        default=[multiprocessing.cpu_count()], help="""Maximum
                        number of jobs to be used. If absent, one job per CPU
                        will be spawned.""")

    parser.add_argument('-n', '--norm', metavar='norm', type=str, nargs=1,
                        help="""Laplacian normalization to be
                        employed. Possibilities are "geig", "unnorm", "rw" and
                        "sym". Defaults to geig for the full brain and ROI analyses, and to unnorm otherwise.""")

    parser.add_argument('-fb', '--full-brain', action='store_true',
                        help="""Calculate full brain feature gradient analysis.""")

    parser.add_argument('-hy', '--hybrid', action='store_true',
                        help="""Calculate searchlight VB index with hybrid approach.""")
    
    parser.add_argument('-rh', '--reho', action='store_true',
                        help="""Calculate the KCC index for ReHo approach.""")
    
    parser.add_argument('-vol', '--volume', action='store_true',
                        help="""Do not map results to surface.""")
    
    parser.add_argument('-ta', '--temporal-analysis', action='store_true',
                        help="""Calculate the time varying VB index of a time window.""")
    
    parser.add_argument('-ws', '--window-size', metavar='integer', type=int, nargs=1, default=10,
                        help="""Window size for Temporal Analysis.""")
    
    parser.add_argument('-st', '--step', metavar='integer', type=int, nargs=1, default=1,
                        help="""Step for Temporal Analysis.""")
    
    parser.add_argument('-sz', '--size', metavar='integer', type=int, nargs=1, default=3,
                        help="""Size for Temporal Analysis.""")
    
    parser.add_argument('-p', '--path', metavar='integer', type=str, nargs=1, default="/tmp/temp_folder/",
                        help="""Path for temporal folder""")
                        
    parser.add_argument('-vm', '--volmask', metavar='file', type=str,
                        nargs=1, default=None, help="""Nifti file containing the whole brain mask
                        in volumetric space. Only relevant if computing the volumetric VB.""")

    parser.add_argument('-m', '--mask', metavar='file', type=str,
                               nargs=1, help="""File containing the labels to
                               identify the cortex, rather than the medial
                               brain structures. This flag must be set for
                               the searchlight and full brain analyses.""")

    parser.add_argument('-c', '--clusters', metavar='file', type=str, nargs=1, default=None,
                        help="""File specifying the surface clusters. The cluster
                        with index 0 is expected to denote the medial brain
                        structures and will be ignored.""")
                        
    parser.add_argument('-t', '--tol', metavar='tolerance', type=float, nargs=1,
                        default=["def_tol"], help="""Residual tolerance (stopping criterion) for LOBPCG. 
                        Default value = sqrt(10e-18)*n, where n is the number of nodes per graph. Note that
                        the LOBPCG algorithm is only utilised for full-brain analysis.""")
                        
    parser.add_argument('-mi', '--maxiter', metavar='max iterations', type=int, nargs=1, default=[50],
                        help="""Maximum number of iterations for LOBPCG. Defaults to 50.""")
    
    parser.add_argument('-debug', '--debug', action='store_true',
                        help="""Save additional files for debugging.""")

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('-s', '--surface', metavar='file', type=str,
                               nargs=1, help="""File containing the surface
                                              mesh.""", required=False)

    requiredNamed.add_argument('-d', '--data', metavar='file', type=str,
                               nargs=1, help="""File containing the data over
                                              the surface (or volume if hybrid).""", required=True)


    requiredNamed.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Base name for the
                                              output files.""", required=True)

    return parser

def main():
    """
    Main function where argument parser and main logic is handled.

    Returns
    -------
    None.

    """
    
    parser = create_parser()
    args = parser.parse_args()

    n_cpus = args.jobs[0]
    
    eigenvalue_index = args.eigenvalue_index    
    

    if not args.volume:
        nib_surf, vertices, faces = vb_index.open_gifti_surf(args.surface[0])

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
            nib_surf.meta['AnatomicalStructurePrimary'] = hemi

    nib = nibabel.load(args.data[0],mmap=False)
    if (args.hybrid or args.volume):
        data = np.array(nib.dataobj)
        affine = nib.affine
        header = nib.header
    else:
        if len(nib.darrays) > 1:
            data = np.array([n.data for n in nib.darrays]).transpose()
        else:
            data = nib.darrays[0].data
            
    if (args.norm is not None and args.norm[0] == 'rw'):
        print('Warning: this method makes use of the Random Walk Normalized Laplacian, and has not been tested rigorously yet.')
    if (args.norm is not None and args.norm[0] == 'sym'):
        print('Warning: this method makes use of the Symmetric Normalized Laplacian, and has not been tested rigorously yet.')
            
    if (args.norm is not None and args.norm[0] == 'rw'):
        print('Warning: this method makes use of the Random Walk Normalized Laplacian, and has not been tested rigorously yet.')
    if (args.norm is not None and args.norm[0] == 'sym'):
        print('Warning: this method makes use of the Symmetric Normalized Laplacian, and has not been tested rigorously yet.')

    if args.full_brain:
        print("Running full brain analysis")
        if args.mask is None:
            sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
            sys.exit(2)
            quit()
        # Read labels
        _, labels = vb_index.open_gifti(args.mask[0])
        cort_index = np.array(labels, bool)
        Z = np.array(cort_index, dtype=int)
        if args.norm is None:
            L_norm = 'geig'
        else:
            L_norm = args.norm[0]
        try:
            vb_index.compute_vb_metrics("vb_cluster", surf_vertices=vertices, surf_faces=faces, n_cpus=n_cpus, data=data, norm=L_norm, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0] + "." + L_norm, nib_surf=nib_surf, cluster_index=Z, full_brain=True, debug=args.debug)
            #full_brain=True
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()
            
    elif args.volume:
        if args.reho:
            print("Running ReHo approach with no surface mapping")
        else:
            print("Running searchlight analysis with no surface mapping")
        if args.mask:
            _, labels = vb_index.open_gifti(args.mask[0])
            cort_index = np.array(labels, bool)
        else:
            cort_index = None
        if args.norm is None:
            L_norm = 'unnorm'
        else:
            L_norm = args.norm[0]
        if args.volmask:
            brain_mask = nibabel.load(args.volmask[0],mmap=False)
            brain_mask = np.array(brain_mask.dataobj)
        else:
            brain_mask = None
        if args.temporal_analysis:
            print("Running Temporal Analysis with no surface mapping")
            if not type(args.window_size) == int:
                window_size = args.window_size[0]
            else:
                window_size = args.window_size
            if not type(args.step) == int:
                steps = args.step[0]
            else:
                steps = args.step
            if not type(args.size) == int:
                size = args.size[0]
            else:
                size = args.size
            try:
                vb_index.compute_temporal_analysis_volumetric(window_size=window_size, steps=steps, size=size, path=args.path, n_cpus=n_cpus, nib=nib, affine=affine, header=header, norm=L_norm, cort_index=cort_index, brain_mask=brain_mask, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0], reho=args.reho, eigenvalue_index=eigenvalue_index)
                sys.exit(1)
            except Exception as error:
                sys.stderr.write(str(error))
                sys.exit(2)
                quit()
        try:
                vb_index.compute_vb_metrics(internal_loop_func="vb_vol", n_cpus=n_cpus, data=data, affine=affine, header=header, norm=L_norm, cort_index=cort_index, brain_mask=brain_mask, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], eigenvalue_index = eigenvalue_index, output_name=args.output[0] + "." + L_norm, reho=args.reho)
            
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()    

    elif args.clusters is None:
        if args.hybrid:
            if args.norm is None:
                L_norm = 'unnorm'
            else:
                L_norm = args.norm[0]
            _, labels = vb_index.open_gifti(args.mask[0])
            cort_index = np.array(labels, bool)
            if args.temporal_analysis:
                print("Running Temporal Analysis with Hybrid Approach")
                if not type(args.window_size) == int:
                    window_size = args.window_size[0]
                else:
                    window_size = args.window_size
                if not type(args.step) == int:
                    steps = args.step[0]
                else:
                    steps = args.step
                if not type(args.size) == int:
                    size = args.size[0]
                else:
                    size = args.size
                try:
                    vb_index.compute_temporal_analysis_hybrid(window_size=window_size, steps=steps, size=size, path=args.path, surf_vertices=vertices, surf_faces=faces, affine=affine, n_cpus=n_cpus, nib=nib, norm=L_norm, cort_index=cort_index, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0], nib_surf=nib_surf, k=3, reho=args.reho, debug=args.debug, eigenvalue_index=eigenvalue_index)
                    sys.exit(1)
                except Exception as error:
                    sys.stderr.write(str(error))
                    sys.exit(2)
                    quit()
            if args.reho:
                print("Running ReHo approach")
            else:
                print("Running searchlight analysis with hybrid approach")
            if args.mask is None:
                sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
                sys.exit(2)
                quit()
          
            # Read labels
            try:
                vb_index.compute_vb_metrics("vb_hybrid", surf_vertices=vertices, surf_faces=faces, affine=affine, n_cpus=n_cpus, data=data, norm=L_norm, cort_index=cort_index, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0], nib_surf=nib_surf, k=3, reho=args.reho, debug=args.debug)
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
            _, labels = vb_index.open_gifti(args.mask[0])
            cort_index = np.array(labels, bool)
            if args.norm is None:
                L_norm = 'unnorm'
            else:
                L_norm = args.norm[0]
            try:
                vb_index.compute_vb_metrics("vb_index", surf_vertices=vertices, surf_faces=faces, n_cpus=n_cpus, data=data, norm=L_norm, cort_index=cort_index, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0] + "." + L_norm, nib_surf=nib_surf)
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
        nib, Z = vb_index.open_gifti(args.clusters[0])
        Z = np.array(Z, dtype=np.int)
        if args.norm is None:
            L_norm = 'geig'
        else:
            L_norm = args.norm[0]
        try:
            vb_index.compute_vb_metrics("vb_cluster", False, vertices, faces, n_cpus, data, L_norm, args.tol[0], args.maxiter[0], args.output[0] + "." + L_norm, nib_surf, k=3, cluster_index=Z, cort_index=cort_index, affine=affine, debug=args.debug)
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()


if __name__ == "__main__":
    main()