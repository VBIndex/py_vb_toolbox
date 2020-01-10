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
                        will be spawned""")

    parser.add_argument('-n', '--norm', metavar='norm', type=str, nargs=1,
                        default=["geig"], help="""Laplacian normalization to be
                        used. Possibilities are "geig", "unnorm", "rw" and
                        "sym". Defaults to geig.""")

    parser.add_argument('-fb', '--full-brain', action='store_true',
                        help="""Calculate full brain feature gradient analyses.""")

    parser.add_argument('-m', '--mask', metavar='file', type=str,
                               nargs=1, help="""File containing the labels to
                               identify the cortex, rather than the medial
                               brain structures. This flag must be set for
                               normal analyses and full brain analyses.""")

    parser.add_argument('-c', '--clusters', metavar='file', type=str, nargs=1, default=None,
                        help="""File containing the surface clusters. Cluster
                        with index 0 are expected to denote the medial brain
                        structures and will be ignored.""")

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('-s', '--surface', metavar='file', type=str,
                               nargs=1, help="""File containing the surface
                                              mesh""", required=True)

    requiredNamed.add_argument('-d', '--data', metavar='file', type=str,
                               nargs=1, help="""File containing the data over
                                              the surface""", required=True)


    requiredNamed.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Base name for the
                                              output files""", required=True)

    return parser


def main():

    parser = create_parser()
    args = parser.parse_args()

    n_cpus = args.jobs[0]
    nib_surf, vertices, faces = io.open_gifti_surf(args.surface[0])
    nib = nibabel.load(args.data[0])
    cifti = nib.darrays[0].data

    if args.full_brain:
        print("Running full brain analyses")
        if args.mask is None:
            print("A mask file must be provided through the --label flag. See --help")
            quit()
        _, labels = io.open_gifti(args.mask[0])
        cort_index = np.array(labels, np.bool)
        Z = np.array(cort_index, dtype=np.int)
        result = vb.vb_cluster(vertices, faces, n_cpus, cifti, Z, args.norm[0], args.output[0] + "." + args.norm[0], nib_surf)

    elif args.clusters is None:
        print("Running searchlight analyses")
        if args.mask is None:
            print("A mask file must be provided through the --label flag. See --help")
            quit()
        # Read labels
        _, labels = io.open_gifti(args.mask[0])
        cort_index = np.array(labels, np.bool)
        result = vb.vb_index(vertices, faces, n_cpus, cifti, args.norm[0], cort_index, args.output[0] + "." + args.norm[0], nib_surf)

    else:
        print("Running ROI analyses")
        nib, Z = io.open_gifti(args.clusters[0])
        Z = np.array(Z, dtype=np.int)
        result = vb.vb_cluster(vertices, faces, n_cpus, cifti, Z, args.norm[0], args.output[0] + "." + args.norm[0], nib_surf)


if __name__ == "__main__":
    main()
