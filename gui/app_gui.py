import tkinter as tk
import webbrowser
from tkinter import filedialog
import os
import sys
import pandas as pd
import subprocess
from tkinter import messagebox
from PIL import Image, ImageTk
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../vb_toolbox')
from app import create_parser


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
        self.pwd = os.path.dirname(__file__)

        self.vb_cmd = tk.StringVar()
        self.vb_cmd.set("")
        self.view_folder = tk.StringVar()
        self.view_folder.set("None")



        # frame info
        self.frame_info = tk.Frame(self.master, width=500)
        self.frame_info.grid(row=0, column=0)
        img = Image.open("icon.png")
        zoom = 0.4
        pixels_x, pixels_y = tuple([int(zoom * x) for x in img.size])
        self.icon = ImageTk.PhotoImage(img.resize((pixels_x, pixels_y)))
        self.label_icon = tk.Label(self.frame_info, image=self.icon)
        self.label_icon.image = self.icon
        self.label_icon.grid()

        tk.Label(self.frame_info, justify=tk.LEFT, wraplength=300, text=self.parser.description).grid(sticky=tk.W, padx=20)
        link_gh = tk.Label(self.frame_info, text='--> Link to GitHub\n', fg="black", cursor="hand2")
        link_gh.grid(sticky=tk.W, padx=20)
        link_gh.bind("<Button-1>", lambda e: webbrowser.open_new('https://github.com/VBIndex/py_vb_toolbox'))

        authors = 'Authors:\nLucas da Costa  Campos (lqccampos@gmail.com)\nand Claude J Bajada (claude.bajada@um.edu.mt)'
        tk.Label(self.frame_info, anchor='w', justify=tk.LEFT, text=authors).grid(sticky=tk.W, padx=20)
        #tk.Label(self.frame_info, text=self.parser.epilog.replace("|n", "\n")).grid()

        tk.Label(self.frame_info, text='\nReferences:').grid(sticky=tk.W, padx=20)
        refs_dict = {'Bajada et al. PsyArXiv, 2020': "https://psyarxiv.com/7vzbk/",
                     'ref2': "http://www.google.com",
                     'ref3': "http://www.google.com"}

        for key, value in refs_dict.items():
            link = tk.Label(self.frame_info, anchor='w', text=key, fg="black", cursor="hand2")
            link.grid(sticky=tk.W, padx=20)
            link.bind("<Button-1>", lambda e: webbrowser.open_new(value))

        # frame settings
        self.frame_run = tk.Frame(self.master)
        self.frame_run.grid(row=0, column=1)

        tk.Label(self.frame_run, anchor='w', justify=tk.LEFT, text='SETTINGS:\n\nRequired arguments:').grid(row=0, column=0, sticky=tk.W)
        tk.Button(self.frame_run, text="Set surface file:", command=lambda: self.fname_to_flag('-s')).grid(row=1, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-s']).grid(row=1, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set data file:", command=lambda: self.fname_to_flag('-d')).grid(row=2, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-d']).grid(row=2, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set output name:", command=self.set_output_name).grid(row=3, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-o']).grid(row=3, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Optional arguments:').grid(row=4, column=0, sticky=tk.W)
        tk.Button(self.frame_run, text="Set mask file:", command=lambda: self.fname_to_flag('-m')).grid(row=5, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-m']).grid(row=5, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set cluster file:", command=lambda: self.fname_to_flag('-c')).grid(row=6, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.var_dict['-c']).grid(row=6, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Jobs:').grid(row=7, column=0, sticky=tk.W)
        tk.Entry(self.frame_run, textvariable=self.var_dict['-j']).grid(row=7, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Full brain analysis:').grid(row=8, column=0, sticky=tk.W)
        tk.Button(self.frame_run, textvariable=self.var_dict['-fb'], command=self.toggle_fb).grid(row=8, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Normalization:').grid(row=9, column=0, sticky=tk.W)
        tk.OptionMenu(self.frame_run, self.var_dict['-n'], "geig", "unnorm", "rw", "sym").grid(row=9, column=1, sticky=tk.W)

        tk.Button(self.frame_run, anchor='n', text='--> Run vb_tool <--', command=self.run_analysis).grid(sticky=tk.N+tk.W)

        tk.Label(self.frame_run, text='Command:').grid(sticky=tk.W)
        tk.Label(self.frame_run, width=20, anchor='w', justify=tk.LEFT, textvariable=self.vb_cmd, bg='white', fg='black').grid(sticky=tk.W)

        # frame view
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
        self.vb_cmd.set(vb_cmd.replace(' -', '\n-'))

        terminal = subprocess.Popen(vb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = terminal.communicate()
        if terminal.returncode != 0:
            messagebox.showinfo("An error occured!", stderr)

    def open_wb_view(self):
        surfs = glob.glob(self.view_folder.get() + '/*.inflated*')
        outputs = glob.glob(self.view_folder.get() + '/*geig*')
        wb_cmd = 'wb_view'
        for surf in surfs:
            wb_cmd = f'{wb_cmd} {surf}'

        for output in outputs:
            wb_cmd = f'{wb_cmd} {output}'

        wb_cmd = f'{wb_cmd} &'
        print(wb_cmd)
        subprocess.Popen(wb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def fname_to_flag(self, flag):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.var_dict[flag].set(fname)

    def set_output_name(self):
        fname = filedialog.asksaveasfilename(initialdir=self.pwd, title="Set an output name")
        if fname:
            self.var_dict['-o'].set(fname)
            self.view_folder.set(fname)

    def toggle_fb(self):
        if self.var_dict['-fb'].get() == 'Yes':
            self.var_dict['-fb'].set('No')
        else:
            self.var_dict['-fb'].set('Yes')

    def set_view_folder(self):
        fname = filedialog.askdirectory(initialdir=self.pwd, title="Select a Folder")
        if fname:
            self.view_folder.set(fname)


if __name__ == "__main__":
    root = tk.Tk()
    vp_toolbox_gui(root)
    root.mainloop()