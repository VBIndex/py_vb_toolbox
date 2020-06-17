import tkinter as tk
from tkinter import ttk
import webbrowser
from tkinter import filedialog
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../vb_toolbox')
from app import create_parser
import pandas as pd
import subprocess
from tkinter import messagebox
from PIL import Image, ImageTk
import glob

class vp_toolbox_gui:
    def __init__(self, master):
        self.master = master
        master.title("VB Toolbox v.1.1.0")

        # default settings:
        self.df_args = pd.read_csv('defaults.csv', delimiter=';')
        for idx, row in self.df_args.iterrows():
            self.df_args.loc[idx, 'strvar'] = tk.StringVar()
            self.df_args.loc[idx, 'strvar'].set(row.default)
        self.pwd = os.path.dirname(__file__)
        self.vb_cmd = tk.StringVar()
        self.vb_cmd.set("")
        self.view_folder = tk.StringVar()
        self.view_folder.set("None")
        self.parser = create_parser()

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
        tk.Button(self.frame_run, text="Set surface file:", command=self.set_surf_fname).grid(row=1, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 's']['strvar'].values[0]).grid(row=1, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set data file:", command=self.set_data_fname).grid(row=2, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'd']['strvar'].values[0]).grid(row=2, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set output name:", command=self.set_output_fname).grid(row=3, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'o']['strvar'].values[0]).grid(row=3, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Optional arguments:').grid(row=4, column=0, sticky=tk.W)
        tk.Button(self.frame_run, text="Set mask file:", command=self.set_mask_fname).grid(row=5, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'm']['strvar'].values[0]).grid(row=5, column=1, sticky=tk.W)
        tk.Button(self.frame_run, text="Set cluster file:", command=self.set_cluster_fname).grid(row=6, column=0, sticky=tk.W)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'c']['strvar'].values[0]).grid(row=6, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Jobs:').grid(row=7, column=0, sticky=tk.W)
        tk.Entry(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'j']['strvar'].values[0]).grid(row=7, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Full brain analysis:').grid(row=8, column=0, sticky=tk.W)
        tk.Button(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'fb']['strvar'].values[0], command=self.toggle_fb).grid(row=8, column=1, sticky=tk.W)

        tk.Label(self.frame_run, text='Normalization:').grid(row=9, column=0, sticky=tk.W)
        tk.OptionMenu(self.frame_run, self.df_args.loc[self.df_args.flag == 'n']['strvar'].values[0], "geig", "unnorm", "rw", "sym").grid(row=9, column=1, sticky=tk.W)

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
        exclude_list = ['None', 'No', 'default']
        vb_cmd = 'vb_tool'
        for ind, arg in self.df_args.iterrows():
            val = self.df_args.loc[self.df_args.flag == arg.flag]['strvar'].values[0].get()
            if val not in exclude_list:
                if arg.flag == 'fb':
                    vb_cmd = f'{vb_cmd} -{arg.flag}'
                else:
                    vb_cmd = f'{vb_cmd} -{arg.flag} {val}'

        print(vb_cmd)
        self.vb_cmd.set(vb_cmd.replace(' -', '\n-'))

        terminal = subprocess.Popen(vb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = terminal.communicate()
        if terminal.returncode != 0:
            messagebox.showinfo("An error occured!", stderr)

    def open_wb_view(self):
        surfs = glob.glob(self.view_folder.get() + '/*inflated*')
        outputs = glob.glob(self.view_folder.get() + '/*vb*')
        wb_cmd = 'wb_view'
        for surf in surfs:
            wb_cmd = f'{wb_cmd} {surf}'

        for output in outputs:
            wb_cmd = f'{wb_cmd} {output}'

        wb_cmd = f'{wb_cmd} &'
        print(wb_cmd)
        subprocess.Popen(wb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)


    def set_surf_fname(self):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.df_args.loc[self.df_args.flag == 's']['strvar'].values[0].set(fname)

    def set_data_fname(self):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.df_args.loc[self.df_args.flag == 'd']['strvar'].values[0].set(fname)

    def set_output_fname(self):
        fname = filedialog.asksaveasfilename(initialdir=self.pwd, title="Set an output name")
        if fname:
            self.df_args.loc[self.df_args.flag == 'o']['strvar'].values[0].set(fname)

    def set_mask_fname(self):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.df_args.loc[self.df_args.flag == 'm']['strvar'].values[0].set(fname)

    def set_cluster_fname(self):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.df_args.loc[self.df_args.flag == 'c']['strvar'].values[0].set(fname)

    def toggle_fb(self):
        toggle_val = self.df_args.loc[self.df_args.flag == 'fb']['strvar'].values[0].get()
        if toggle_val == 'Yes':
            self.df_args.loc[self.df_args.flag == 'fb']['strvar'].values[0].set('No')
        else:
            self.df_args.loc[self.df_args.flag == 'fb']['strvar'].values[0].set('Yes')

    def set_view_folder(self):
        fname = filedialog.askdirectory(initialdir=self.pwd, title="Select a Folder")
        if fname:
            self.view_folder.set(fname)


if __name__ == "__main__":
    root = tk.Tk()
    vp_toolbox_gui(root)
    root.mainloop()