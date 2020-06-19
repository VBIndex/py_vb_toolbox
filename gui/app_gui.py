#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Nicole Eichert <n.eichert@googlemail.com>
#
# Distributed under terms of the GNU license.

import glob
import subprocess
import sys
import os
import textwrap
import tkinter as tk
import webbrowser
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../vb_toolbox')
from app import create_parser

# TODO: add comments
# TODO: add to wishlist
# TODO: pull request
# TODO: authors
# TODO: labels
# TODO: surface in viewer
# TODO: version from setup.py
# TODO: make video
# TODO: pull from new version upgrade pip
# TODO: helpbox to tooltip


class vp_toolbox_gui:
    def __init__(self, master):
        self.master = master
        master.title("VB Toolbox v.1.1.0")

        # get info about arguments from parser
        self.parser = create_parser()
        self.args = self.parser.__dict__["_option_string_actions"]
        flags = list(self.args.keys())
        flags = [x for x in flags if not '--' in x]
        self.var_dict = dict((el, tk.StringVar()) for el in flags)
        for flag, var in self.var_dict.items():
            if isinstance(self.args[flag].default, list):
                var.set(str(self.args[flag].default[0]))
            else:
                var.set(str(self.args[flag].default))
        self.wd = os.path.dirname(__file__)

        # Some Defaults
        self.display_cmd = tk.StringVar()
        self.display_cmd.set("")
        self.view_folder = tk.StringVar()
        self.view_folder.set("None")

        # frame containing app info
        self.frame_info = tk.Frame(self.master, width=500)
        self.frame_info.grid(row=0, column=0)
        img = Image.open(os.path.join(self.wd, "icon.png"))
        zoom = 0.3
        pixels_x, pixels_y = tuple([int(zoom * x) for x in img.size])
        self.icon = ImageTk.PhotoImage(img.resize((pixels_x, pixels_y)))
        self.label_icon = tk.Label(self.frame_info, image=self.icon)
        self.label_icon.image = self.icon
        self.label_icon.grid()

        tk.Label(self.frame_info, justify=tk.LEFT, wraplength=300, text='Calculate the Vogt-Bailey index of a dataset.').grid(sticky=tk.W, padx=20)
        refs_dict = {'-> Bajada et al. PsyArXiv, 2020': "https://psyarxiv.com/7vzbk/"}
        for key, value in refs_dict.items():
            link = tk.Label(self.frame_info, anchor='w', text=key, fg="black", cursor="hand2")
            link.grid(sticky=tk.W, padx=20)
            link.bind("<Button-1>", lambda e: webbrowser.open_new(value))
        tk.Button(self.frame_info, text='About ...', command=self.show_about).grid(sticky=tk.W, padx=20)
        tk.Button(self.frame_info, text='GitHub ...', command=lambda: webbrowser.open_new('https://github.com/VBIndex/py_vb_toolbox')).grid(sticky=tk.W, padx=20)




        # frame containing settings
        self.frame_run = tk.Frame(self.master)
        self.frame_run.grid(row=0, column=1)

        tk.Label(self.frame_run, anchor='w', justify=tk.LEFT, text='SETTINGS:\n\nRequired arguments:').grid(row=0, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set surface file:", command=lambda: self.fname_to_flag('-s')).grid(row=1, column=1, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-s']).grid(row=1, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-s')).grid(row=1, column=0)

        tk.Button(self.frame_run, text="Set data file:", command=lambda: self.fname_to_flag('-d')).grid(row=2, column=1, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-d']).grid(row=2, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-d')).grid(row=2, column=0)

        tk.Button(self.frame_run, text="Change folder:", command=self.set_output_name).grid(row=3, column=1, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-o']).grid(row=3, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-o')).grid(row=3, column=0)

        tk.Label(self.frame_run, text='\nOptional arguments:').grid(row=4, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set mask file:", command=lambda: self.fname_to_flag('-m')).grid(row=5, column=1, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-m']).grid(row=5, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-m')).grid(row=5, column=0)

        tk.Button(self.frame_run, text="Set cluster file:", command=lambda: self.fname_to_flag('-c')).grid(row=6, column=1, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-c']).grid(row=6, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-c')).grid(row=6, column=0)

        tk.Label(self.frame_run, text='Jobs:').grid(row=7, column=1, sticky=tk.W)
        tk.Entry(self.frame_run, width=15, textvariable=self.var_dict['-j']).grid(row=7, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-j')).grid(row=7, column=0)

        tk.Label(self.frame_run, text='Full brain analysis:').grid(row=8, column=1, sticky=tk.W)
        tk.Button(self.frame_run, textvariable=self.var_dict['-fb'], command=self.toggle_fb).grid(row=8, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-fb')).grid(row=8, column=0)

        tk.Label(self.frame_run, text='Normalization:').grid(row=9, column=1, sticky=tk.W)
        tk.OptionMenu(self.frame_run, self.var_dict['-n'], "geig", "unnorm", "rw", "sym").grid(row=9, column=2, sticky=tk.W)
        tk.Button(self.frame_run, text="?", command=lambda: self.show_help('-n')).grid(row=9, column=0)

        tk.Button(self.frame_run, anchor='n', text='--> Run vb_tool <--', pady=10, fg='red', command=self.run_analysis).grid(row=10, column=1, sticky=tk.N+tk.W)

        tk.Label(self.frame_run, text='Command:').grid(row=11, column=1, sticky=tk.W)
        tk.Label(self.frame_run, width=15, anchor='w', justify=tk.LEFT, textvariable=self.display_cmd, bg='white', fg='black').grid(row=12, column=1, sticky=tk.W)

        # frame for view options
        self.frame_view = tk.Frame(self.master)
        self.frame_view.grid(row=0, column=2, sticky=tk.W)
        tk.Label(self.frame_view, anchor='w', justify=tk.LEFT, text='VIEW RESULTS:').grid(sticky=tk.W)
        tk.Button(self.frame_view, text='Set output folder:', command=self.set_view_folder).grid(sticky=tk.W)
        tk.Label(self.frame_view, anchor='w', justify=tk.LEFT, textvariable=self.view_folder).grid(sticky=tk.W)
        tk.Button(self.frame_view, text='Open wb_view', command=self.open_wb_view).grid(sticky=tk.W)

    def run_analysis(self):
        exclude_list = ['None', 'False', '==SUPPRESS==']
        vb_cmd = 'vb_tool'
        for flag, var in self.var_dict.items():
            val = var.get()
            if val not in exclude_list:
                if self.args[flag].nargs == 0:
                    vb_cmd = f'{vb_cmd} {flag}'
                else:
                    vb_cmd = f'{vb_cmd} {flag} {val}'

        print(vb_cmd)
        self.display_cmd.set(vb_cmd.replace(' -', '\n-'))

        terminal = subprocess.Popen(vb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = terminal.communicate()
        if terminal.returncode != 0:
            messagebox.showinfo("An error occured!", stderr)

    def open_wb_view(self):
        surfs = glob.glob(self.view_folder.get() + '/*.inflated*')
        search_str = self.var_dict['-n'].get() + '.vb'
        outputs = glob.glob(self.view_folder.get() + f'/*{search_str}*')
        wb_cmd = 'wb_view'
        for surf in surfs:
            wb_cmd = f'{wb_cmd} {surf}'
        for output in outputs:
            wb_cmd = f'{wb_cmd} {output}'

        wb_cmd = f'{wb_cmd} &'
        print(wb_cmd)
        try:
            subprocess.Popen(wb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except:
            messagebox.showinfo("An error occured!", 'wb_view can not be started. For more information:\nhttps://www.humanconnectome.org/software/workbench-command')

    def fname_to_flag(self, flag):
        fname = filedialog.askopenfilename(initialdir=self.wd, title="Select a File")
        if fname:
            self.var_dict[flag].set(fname)

    def set_output_name(self):
        fname = filedialog.asksaveasfilename(initialdir=self.wd, title="Set an output name")
        if fname:
            self.var_dict['-o'].set(fname)
            self.view_folder.set(fname)

    def toggle_fb(self):
        if self.var_dict['-fb'].get() == 'Yes':
            self.var_dict['-fb'].set('No')
        else:
            self.var_dict['-fb'].set('Yes')

    def set_view_folder(self):
        fname = filedialog.askdirectory(initialdir=self.wd, title="Select a Folder")
        if fname:
            self.view_folder.set(fname)

    def show_help(self, flag):
        help_msg = textwrap.shorten(self.args[flag].help, width=200)
        messagebox.showinfo("Argument settings", help_msg)

    def show_about(self):
        about_msg = textwrap.shorten(textwrap.dedent(self.parser.epilog), width=1000, drop_whitespace=False).replace('|n', '\n\n')
        messagebox.showinfo("About vb_tool", about_msg)



def main():
    root = tk.Tk()
    vp_toolbox_gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()