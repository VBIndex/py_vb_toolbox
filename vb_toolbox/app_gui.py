#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Nicole Eichert <n.eichert@googlemail.com>, Keith George Ciantar <keithgeorge98@gmail.com>
#
# Distributed under terms of the GNU license.

import glob
import subprocess
import sys
import os
import textwrap
import webbrowser

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.scrolledtext as scrolledtext

from vb_toolbox.app import create_parser

import pkg_resources
import threading
import queue

class VPToolboxGui:
    def __init__(self, master):
        # Get the root tkinter object
        self.master = master
        self.master.resizable(width=False, height=False)

        # Get the version number from the installed vb_toolbox
        version = pkg_resources.require('vb_toolbox')[0].version
        self.master.title(f'VB Toolbox v{version}')

        # Set the weight of the rows and columns
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

        # Get info about arguments from parser object in app.py
        self.parser = create_parser()
        self.args = self.parser.__dict__['_option_string_actions']

        flags = list(self.args.keys())
        flags = [x for x in flags if not '--' in x]

        self.var_dict = dict((el, tk.StringVar()) for el in flags)

        for flag, var in self.var_dict.items():
            if isinstance(self.args[flag].default, list):
                var.set(str(self.args[flag].default[0]))
            else:
                var.set(str(self.args[flag].default))

        # Location of gui script
        self.sd = os.path.dirname(__file__)

        # Users working directory
        self.wd = os.getcwd()

        # Set the analysis type
        self.analysis = tk.StringVar()
        self.analysis.set('Searchlight')
        self.var_dict['-c'].set('None')
        self.var_dict['-m'].set('')
        self.var_dict['-fb'].set('False')

        # Set the run button string
        self.run_button_text = tk.StringVar()
        self.run_button_text.set('Run vb_tool')

        # Initialize some defaults
        self.display_cmd = tk.StringVar()
        self.display_cmd.set('')

        self.view_folder = tk.StringVar()
        self.view_folder.set('None')

        self.view_surf = tk.StringVar()
        self.view_surf.set('None')

        # Set the colours for each frame
        info_colour = 'burlywood3'
        settings_colour = 'burlywood2'
        results_colour = 'burlywood3'

        # Set the padding for the elements
        self.outer_padding = 10
        self.inner_padding = 3

        # Set the wraplength for the headers
        self.wrap_length = 400

        if self.master.winfo_screenwidth() > 1920:
            self.wrap_length = 550

        # Construct frame containing app info
        self.frame_info = tk.Frame(self.master, padx=self.outer_padding, pady=self.outer_padding, bg=info_colour)
        self.frame_info.grid(row=0, column=0, sticky='news')

        # Set a title with the name of the app
        self.info_title = tk.Label(self.frame_info, justify=tk.CENTER, wraplength=self.wrap_length, text='Vogt-Bailey Toolbox', bg=info_colour)
        self.info_title.config(font=('Roboto', 40))
        self.info_title.grid(sticky=tk.W+tk.E+tk.E, padx=self.outer_padding, pady=self.outer_padding)

        # Place icon in frame
        img = Image.open(os.path.join(self.sd, 'assets/vb_gui_icon.png'))
        zoom = 0.3
        pixels_x, pixels_y = tuple([int(zoom * x) for x in img.size])

        self.icon = ImageTk.PhotoImage(img.resize((pixels_x, pixels_y)))
        self.label_icon = tk.Label(self.frame_info, image=self.icon, bg=info_colour)
        self.label_icon.image = self.icon
        self.label_icon.grid()

        # Application info
        tk.Label(self.frame_info, justify=tk.LEFT, wraplength=self.wrap_length, text='Calculate the Vogt-Bailey (VB) index of a dataset by choosing from the three main analysis types.', bg=info_colour).grid(sticky=tk.W+tk.E, padx=self.outer_padding, pady=self.outer_padding/2)

        # TODO: add more references to dictionary
        refs_dict = {'Bajada et al. (2020), NeuroImage\nDOI: 10.1016/j.neuroimage.2020.117140': 'https://doi.org/10.1016/j.neuroimage.2020.117140'}

        for key, value in refs_dict.items():
            link = tk.Label(self.frame_info, justify=tk.LEFT,  wraplength=self.wrap_length, text=key, fg='black', cursor='hand2', bg=info_colour)
            link.grid(sticky=tk.W, padx=self.outer_padding, pady=self.outer_padding/2)
            link.bind('<Button-1>', lambda e: webbrowser.open_new(value))

        # Quick start button
        tk.Button(self.frame_info, text='Quick Start', command=self.show_quick_start).grid(sticky=tk.W+tk.E+tk.E, padx=self.outer_padding, pady=self.outer_padding/2)

        # About button
        tk.Button(self.frame_info, text='About', command=self.show_about).grid(sticky=tk.W+tk.E+tk.E, padx=self.outer_padding, pady=self.outer_padding/2)

        # Github button
        tk.Button(self.frame_info, text='GitHub', command=lambda: webbrowser.open_new('https://github.com/VBIndex/py_vb_toolbox')).grid(sticky=tk.W+tk.E+tk.E, padx=self.outer_padding, pady=self.outer_padding/2)

        # Construct frame containing settings
        self.frame_run = tk.Frame(self.master, padx=self.outer_padding, pady=self.outer_padding, bg=settings_colour)
        self.frame_run.grid(row=0, column=1, sticky='news')

        # Declare the command display so that it can be updated by the other variables
        self.cmd_display = scrolledtext.ScrolledText(self.frame_run, height=7, width=35)
        self.cmd_display.grid(row=12, column=0, columnspan=3, sticky=tk.W+tk.E, padx=self.outer_padding, pady=self.inner_padding)

        # Set the box width
        bw = 15

        # Settings header
        self.settings_label = tk.Label(self.frame_run, justify=tk.CENTER, wraplength=self.wrap_length, text='Settings', bg=settings_colour)
        self.settings_label.config(font=('Roboto', 24))
        self.settings_label.grid(row=0, column=0, columnspan=3, padx=self.outer_padding, pady=self.outer_padding)

        # Required arguments sub-header
        self.required_label = tk.Label(self.frame_run, justify=tk.CENTER, wraplength=self.wrap_length, text='Required arguments', bg=settings_colour)
        self.required_label.config(font=('TkDefaultFont', 10, 'bold'))
        self.required_label.grid(row=1, column=0, columnspan=3, padx=self.outer_padding, pady=(0, self.inner_padding), sticky=tk.W+tk.E)

        # -s
        # Set the surface file
        tk.Button(self.frame_run, text='Set surface file', width=bw, command=lambda: self.fname_to_flag('-s')).grid(row=2, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        self.surface_entry = tk.Entry(self.frame_run, textvariable=self.var_dict['-s'])
        self.surface_entry.bind('<KeyRelease>', self.update_on_key_release)
        self.surface_entry.grid(row=2, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-s')).grid(row=2, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

        # -d
        # Set the data file
        tk.Button(self.frame_run, text='Set data file', width=bw, command=lambda: self.fname_to_flag('-d')).grid(row=3, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        self.data_entry = tk.Entry(self.frame_run, textvariable=self.var_dict['-d'])
        self.data_entry.bind('<KeyRelease>', self.update_on_key_release)
        self.data_entry.grid(row=3, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-d')).grid(row=3, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

        # -o
        # Set the output file/folder name
        tk.Button(self.frame_run, text='Output name', width=bw, command=self.set_output_name).grid(row=4, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        self.output_entry = tk.Entry(self.frame_run, textvariable=self.var_dict['-o'])
        self.output_entry.bind('<KeyRelease>', self.update_on_key_release)
        self.output_entry.grid(row=4, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-o')).grid(row=4, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

        # Optional arguments sub-header
        self.optional_label = tk.Label(self.frame_run, justify=tk.CENTER, wraplength=self.wrap_length, text='Optional arguments', bg=settings_colour)
        self.optional_label.config(font=('TkDefaultFont', 10, 'bold'))
        self.optional_label.grid(row=5, column=0, sticky=tk.W+tk.E, columnspan=3, padx=self.outer_padding, pady=(self.outer_padding, self.inner_padding))

        # -fb
        # Analysis type
        tk.Label(self.frame_run, text='Analysis type', width=bw, anchor='center').grid(row=6, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        tk.OptionMenu(self.frame_run, self.analysis, 'Searchlight', 'Clustered', 'Full brain', command=self.analysis_type).grid(row=6, column=2, sticky=tk.W+tk.E+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-fb')).grid(row=6, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

        # -m
        # Set the mask file
        self.mask_file_btn = tk.Button(self.frame_run, text='Set mask file', width=bw, command=lambda: self.fname_to_flag('-m'))
        self.mask_file_entry = tk.Entry(self.frame_run, textvariable=self.var_dict['-m'])
        self.mask_file_entry.bind('<KeyRelease>', self.update_on_key_release)
        self.mask_file_qst_btn = tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-m'))

        # Buttons and entry for the mask file are shown by default
        self.show_mask_file_input()

        # -c
        # Set the cluster file
        self.clst_file_btn = tk.Button(self.frame_run, text='Set cluster file', width=bw, command=lambda: self.fname_to_flag('-c'))
        self.clst_file_entry = tk.Entry(self.frame_run, textvariable=self.var_dict['-c'])
        self.clst_file_entry.bind('<KeyRelease>', self.update_on_key_release)
        self.clst_file_qst_btn = tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-c'))

        # -j
        # Set the number of parallel jobs
        tk.Label(self.frame_run, text='Jobs', width=bw, anchor='center').grid(row=9, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        self.jobs_entry = tk.Entry(self.frame_run, textvariable=self.var_dict['-j'])
        self.jobs_entry.bind('<KeyRelease>', self.update_on_key_release)
        self.jobs_entry.grid(row=9, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-j')).grid(row=9, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

        # -n
        # Set the normalisation options
        tk.Label(self.frame_run, text='Normalization', width=bw, anchor='center').grid(row=10, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        tk.OptionMenu(self.frame_run, self.var_dict['-n'], 'geig', 'unnorm', 'rw', 'sym', command=self.update_on_choose).grid(row=10, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        tk.Button(self.frame_run, text='?', command=lambda: self.show_help('-n')).grid(row=10, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

        # Required arguments sub-header
        self.command_label = tk.Label(self.frame_run, justify=tk.CENTER, wraplength=self.wrap_length, text='Command', bg=settings_colour)
        self.command_label.config(font=('TkDefaultFont', 10, 'bold'))
        self.command_label.grid(row=11, column=0, sticky=tk.W+tk.E, columnspan=3, padx=self.outer_padding, pady=(self.outer_padding, self.inner_padding))

        # Run button
        self.run_button = tk.Button(self.frame_run, textvariable=self.run_button_text, pady=self.outer_padding/2, fg='red', command=self.run_analysis)
        self.run_button.config(font=('TkDefaultFont', 10, 'bold'))
        self.run_button.grid(row=13, column=0, columnspan=3, sticky=tk.W+tk.E, padx=self.outer_padding, pady=self.outer_padding/2)

        # Construct frame for viewing options (use info from -s and -o as defaults)
        self.frame_view = tk.Frame(self.master, padx=self.outer_padding, pady=self.outer_padding, bg=results_colour)
        self.frame_view.grid(row=0, column=2, sticky='news')

        # Visualisation header
        self.results_label = tk.Label(self.frame_view, justify=tk.CENTER, wraplength=self.wrap_length, text='Visualisation', bg=results_colour)
        self.results_label.config(font=('Roboto', 24))
        self.results_label.grid(row=0, column=0, columnspan=3, padx=self.inner_padding, pady=self.outer_padding)

        # Results options
        self.empty_label = tk.Label(self.frame_view, justify=tk.CENTER, wraplength=self.wrap_length, text='Results options', bg=results_colour)
        self.empty_label.config(font=('TkDefaultFont', 10, 'bold'))
        self.empty_label.grid(row=1, column=0, columnspan=3, padx=self.outer_padding, pady=(0, self.inner_padding), sticky=tk.W+tk.E)

        # Change surface
        tk.Button(self.frame_view, text='Change surface', width=bw, command=self.set_view_surf).grid(row=2, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)
        tk.Entry(self.frame_view, textvariable=self.view_surf).grid(row=2, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)

        # Change data folder
        tk.Button(self.frame_view, text='Change data folder', width=bw, command=self.set_view_folder).grid(row=3, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)
        tk.Entry(self.frame_view, textvariable=self.view_folder).grid(row=3, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)

        # Open wb_view
        tk.Button(self.frame_view, text='Open wb_view', command=self.open_wb_view).grid(row=4, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)

    # Update the command when a new option is chosen
    def update_on_choose(self, value):
        self.update_command()

    # Update the command when an entry is modified
    def update_on_key_release(self, event):
        self.update_command()

    def update_command(self):
        # Construct command
        exclude_list = ['None', 'False', '==SUPPRESS==']
        vb_cmd = 'vb_tool'

        for flag, var in self.var_dict.items():
            val = var.get()
            if val not in exclude_list:
                if self.args[flag].nargs == 0:
                    vb_cmd = f'{vb_cmd} {flag}'
                else:
                    vb_cmd = f'{vb_cmd} {flag} {val}'

        # Display command in text field
        self.cmd_display.delete('1.0', tk.END)
        self.cmd_display.insert(tk.END, vb_cmd.replace(' -', '\n-'))

        return vb_cmd

    # Run button logic
    def run_analysis(self):
        vb_cmd = self.update_command()

        print(vb_cmd)

        # Change the button text/colour while running
        self.run_button_text.set('Running...')
        self.run_button.config(fg='blue')

        # Run the command in a subprocess
        self.process = subprocess.Popen(vb_cmd, stderr=subprocess.PIPE, shell=True)

        # Read any stderr output from the process using a thread and store them in a queue
        self.q = queue.Queue(maxsize = 1024)
        t = threading.Thread(target=self.reader_thread, args=[self.q])
        t.daemon = True
        t.start()

        # Update the GUI
        self.update(self.q)

    # Source: https://stackoverflow.com/questions/50449082/tkinter-updating-gui-from-subprocess-output-in-realtime
    # Read the outputs from the process
    def reader_thread(self, q):
        try:
            with self.process.stderr as pipe:
                for line in iter(pipe.read, b''):
                    q.put(line)
        finally:
            q.put(None)

    # Update the GUI depending on the state of the process and the stderr output
    def update(self, q):
        for line in self.iter_except(q.get_nowait, queue.Empty):
            if line is None:
                if self.process.poll() is not None:
                    self.run_button_text.set('Run vb_tool')
                    self.run_button.config(fg='red')
                    messagebox.showinfo('Done', 'The process has finished!')
                else:
                    break
            else:
                if self.process.poll() is not None:
                    if self.process.returncode != 0:
                        self.run_button_text.set('Run vb_tool')
                        self.run_button.config(fg='red')
                        messagebox.showinfo('An error occurred!', line)
                    return
                else:
                    break
        self.master.after(10, self.update, q)

    # Helper function to iterate through the queue
    def iter_except(self, function, exception):
        try:
            while True:
                yield function()
        except exception:
            return

    # View button logic
    def open_wb_view(self):
        my_surf = self.view_surf.get()

        # Derive surface from filename
        hemi = 'L' if '.L.' in my_surf else 'R'

        # Select output files based on 'norm' setting
        # Note: the normalisation has the be same the one that needs to be loaded
        # Example: if geig is chosen, wb_view will show any files with gieg in them
        search_str = self.var_dict['-n'].get()
        outputs = glob.glob(self.view_folder.get() + f'/*{search_str}*')
        outputs = [output.replace('\\', '/') for output in outputs]

        # Construct command
        wb_cmd = 'wb_view'
        wb_cmd = f'{wb_cmd} {my_surf}'

        for output in outputs:
            wb_cmd = f'{wb_cmd} {output}'

        wb_cmd = f'{wb_cmd}'

        print(wb_cmd)

        # Call wb_view
        try:
            subprocess.Popen(wb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except:
            messagebox.showinfo('An error occurred!', 'wb_view can not be started. For more information:\nhttps://www.humanconnectome.org/software/workbench-command')

    # Assign a filename to a flag based on button
    def fname_to_flag(self, flag):
        fname = filedialog.askopenfilename(initialdir=self.wd, title='Select a File')
        if fname:
            self.var_dict[flag].set(fname)
            if flag == '-s':
                self.view_surf.set(fname)

        self.update_command()

    # Assign the output filename to a flag
    def set_output_name(self):
        fname = filedialog.asksaveasfilename(initialdir=self.wd, title='Set an output name')
        if fname:
            self.var_dict['-o'].set(fname)
            self.view_folder.set(os.path.dirname(fname))

        self.update_command()

    # Open the option to select a file
    def set_view_surf(self):
        fname = filedialog.askopenfilename(initialdir=self.wd, title='Select a File')
        if fname:
            self.view_surf.set(fname)

    # Open the option to select a folder
    def set_view_folder(self):
        fname = filedialog.askdirectory(initialdir=self.sd, title='Select a Folder')
        if fname:
            self.view_folder.set(fname)

    # Display help from parser when '?' button is pressed
    def show_help(self, flag):
        help_msg = textwrap.shorten(self.args[flag].help, width=200)
        help_msg = ', '.join(self.args[flag].option_strings) + ' :\n' + help_msg
        messagebox.showinfo('Argument settings', help_msg)

    # Display a small tutorial, when 'Quick Start...' is pressed
    def show_quick_start(self):
        about_msg = textwrap.shorten(textwrap.dedent('''To operate this tool follow these steps:|n
        1. Go to the Settings column.|n
        2. Choose your input files, output location and output filename.|n
        3. Determine the type of analysis which will be carried out.|n
        4. Set the mask/cluster file depending on the job.|n
        5. Set the number of parallel jobs spawned by the process.|n
        6. Set the desired normalisation.|n
        7. Run the tool and wait for a pop-up to show the completion.|n
        8. Once the tool has finished you can go to the Visualisations tab to view the models. Note: This requires the Connectome Workbench to be installed on your PC.'''), width=1000, drop_whitespace=False).replace('|n ', '\n\n')
        messagebox.showinfo('Quick Start', about_msg)

    # Display epilog from parser, when 'About...' is pressed
    def show_about(self):
        about_msg = textwrap.shorten(textwrap.dedent(self.parser.epilog), width=1000, drop_whitespace=False).replace('|n ', '\n\n')
        messagebox.showinfo('About vb_tool', about_msg)

    # Show the 'Set mask file' button on the grid
    def show_mask_file_input(self):
        self.mask_file_btn.grid(row=7, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        self.mask_file_entry.grid(row=7, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        self.mask_file_qst_btn.grid(row=7, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

    # Remove the 'Set mask file' button on the grid
    def remove_mask_file_input(self):
        self.mask_file_btn.grid_remove()
        self.mask_file_entry.grid_remove()
        self.mask_file_qst_btn.grid_remove()

    # Show the 'Set cluster file' button on the grid
    def show_cluster_file_input(self):
        self.clst_file_btn.grid(row=8, column=1, sticky=tk.W+tk.E, padx=self.inner_padding, pady=self.inner_padding)
        self.clst_file_entry.grid(row=8, column=2, sticky=tk.W+tk.E, padx=(self.inner_padding, self.outer_padding), pady=self.inner_padding)
        self.clst_file_qst_btn.grid(row=8, column=0, sticky=tk.W+tk.E, padx=(self.outer_padding, self.inner_padding), pady=self.inner_padding)

    # Remove the 'Set cluster file' button on the grid
    def remove_cluster_file_input(self):
        self.clst_file_btn.grid_remove()
        self.clst_file_entry.grid_remove()
        self.clst_file_qst_btn.grid_remove()

    # Choose between 3 types of analysis
    # and show either the mask file option or the cluster file option
    # Set the flags according to the chosen option
    def analysis_type(self, value):
        if self.analysis.get() == 'Clustered':
            self.show_cluster_file_input()
            self.remove_mask_file_input()
            self.var_dict['-c'].set('')
            self.var_dict['-m'].set('None')
            self.var_dict['-fb'].set('False')

        elif self.analysis.get() == 'Full brain':
            self.show_mask_file_input()
            self.remove_cluster_file_input()
            self.var_dict['-c'].set('None')
            self.var_dict['-m'].set('')
            self.var_dict['-fb'].set('True')

        elif self.analysis.get() == 'Searchlight':
            self.show_mask_file_input()
            self.remove_cluster_file_input()
            self.var_dict['-c'].set('None')
            self.var_dict['-m'].set('')
            self.var_dict['-fb'].set('False')

        self.update_command()


def main():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()

    if screen_width > 1920 and screen_width <= 2560:
        root.tk.call('tk', 'scaling', 2.0)
    elif screen_width > 2560:
        root.tk.call('tk', 'scaling', 2.5)

    VPToolboxGui(root)
    root.mainloop()


if __name__ == '__main__':
    main()
