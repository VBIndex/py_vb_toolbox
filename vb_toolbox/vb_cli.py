#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 VB Index Team
#
# Created by The BOBLab
#
# Distributed under terms of the GNU license.

from . import vol_pipe

import argparse
import multiprocessing
import nibabel
import numpy as np
import textwrap as _textwrap
import sys

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
        Ciantar, K. G., Farrugia, C., Galdi, P., Scerri, K., Xu, T., & Bajada, C. J. (2022). Geometric effects of volume-to-surface mapping of fMRI data. Brain Structure and Function, 227(7), 2457-2464.|n
        Farrugia, C., Galdi, P., Irazu, I. A., Scerri, K., & Bajada, C. J. (2024). Local gradient analysis of human brain function using the Vogt-Bailey Index. Brain Structure and Function, 229(2), 497-512.|n
        Galea, K., Escudero, A. A., Montalto, N. A., Vella, N., Smith, R. E., Farrugia, C., ... & Bajada, C. J. (2025). Testing the Vogt-Bailey Index using task-based fMRI across pulse sequence protocols. bioRxiv, 2025-02.
    '''
    
    parser = argparse.ArgumentParser(description='Calculate the Vogt-Bailey Index of a dataset. For more information, refer to https://github.com/VBIndex/py_vb_toolbox.',
                                     epilog=authors + " |n " + references + " |n " + copyright,
                                     formatter_class=MultilineFormatter)
    
    # Added subparsers for each of the different method for VB calculation
    subparsers = parser.add_subparsers(dest='method', 
                                       title="VB method",
                                       required=True, 
                                       help='Different methods for computing the VB Index')
    
    # Subparser for the volumetric analysis
    volumetric = subparsers.add_parser('volumetric', 
                                   help='Computes the VB Index on volumetric data using a searchlight approach')

    # Adding custom arguments for the volumetric analysis
    volumetric.add_argument('-d', '--data', 
                        metavar='file', 
                        type=str,
                        help="""NIfTI file containing the volumetric fMRI data.""", 
                        required=True)

    volumetric.add_argument('-o', '--output', 
                        metavar='file', 
                        type=str,
                        help="""Base name for the output files.""", 
                        required=True)
    
    volumetric.add_argument('-vm', '--volmask', 
                        metavar='file', 
                        type=str,
                        default=None, 
                        help="""NIfTI file containing the whole brain mask in volumetric space.""")
    
    volumetric.add_argument('-ln', '--little-neighbourhood', 
                        action='store_true',
                        help="""Change the amount of voxels taken to create the neighbourhoods from 27 to 7.""")
    
    volumetric.add_argument('-rh', '--reho', 
                        action='store_true',
                        help="""Calculate the KCC index for ReHo approach.""")
    
    volumetric.add_argument('-n', '--norm', 
                        action='store_true',
                        help="""Laplacian normalization to be employed. If this flag is used the "geig" method is employed. Else, it defaults to "unnorm".""")

    volumetric.add_argument('-j', '--jobs', 
                        metavar='N', 
                        type=int,
                        default=multiprocessing.cpu_count(), 
                        help="""Maximum number of jobs to be used. If absent, one job per CPU will be spawned.""")
       
    return parser, volumetric

def main():
    """
    Main function where argument parser and main logic is handled.

    Returns
    -------
    None.

    """
    
    parser, volumetric = create_parser()
    
    # Show help if no arguments are given
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Show help for the volumetric if user called volumetric but no additional arguments
    if len(sys.argv) == 2 and sys.argv[1] == "volumetric":
        print("hi")
        volumetric.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()

    if args.method == 'volumetric':
        n_cpus = args.jobs
        nib = nibabel.load(args.data,mmap=False)
        data = np.array(nib.dataobj)
        affine = nib.affine
        header = nib.header
        
        if args.norm:
            L_norm = "geig"
        else:
            L_norm = "unnorm"
                        
        if args.reho:
            print("Running ReHo approach with no surface mapping")
        else:
            print("Running searchlight analysis with no surface mapping")
            
        if args.volmask:
            brain_mask = nibabel.load(args.volmask,mmap=False)
            brain_mask = np.array(brain_mask.dataobj)
        else:
            brain_mask = None
            
        try:
            vol_pipe.main(n_cpus, data, affine, header, L_norm, brain_mask, output_name=args.output + "." + L_norm, six_f=args.little_neighbourhood, reho=args.reho)

        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()    

if __name__ == "__main__":
    main()