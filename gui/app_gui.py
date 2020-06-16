import tkinter as tk
from tkinter import ttk
import webbrowser
from tkinter import filedialog
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../vb_toolbox')
from app import create_parser
import pandas as pd


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
        self.vb_cmd = ""
        self.surf_fname = tk.StringVar()
        self.surf_fname.set("no file selected")
        self.data_fname = tk.StringVar()
        self.data_fname.set("no file selected")
        self.output_fname = tk.StringVar()
        self.output_fname.set("no output name selected")
        self.parser = create_parser()
        self.jobs = tk.StringVar()
        self.jobs.set('default')

        # frame top
        self.frame_info = tk.Frame(self.master)
        self.frame_info.grid(row=0, column=0)
        self.icon = tk.PhotoImage(file="icon.png")
        self.label_icon = tk.Label(self.frame_info, image=self.icon)
        self.label_icon.image = self.icon
        self.label_icon.grid(row=0, column=0)
        self.frame_info_refs = tk.Frame(self.frame_info)
        self.frame_info_refs.grid(row=0, column=1)

        tk.Label(self.frame_info_refs, anchor='w', text=self.parser.description).grid()
        link_gh = tk.Label(self.frame_info_refs, text='--> Link to GitHub\n', fg="black", cursor="hand2")
        link_gh.grid()
        link_gh.bind("<Button-1>", lambda e: webbrowser.open_new('https://github.com/VBIndex/py_vb_toolbox'))

        tk.Label(self.frame_info_refs, anchor='w', text='Authors:\nLucas da Costa (lqccampos@gmail.com)\n Claude J Bajada (claude.bajada@um.edu.mt)').grid()

        tk.Label(self.frame_info_refs, text='References:').grid()

        refs_dict = {'Bajada et al. PsyArXiv, 2020': "https://psyarxiv.com/7vzbk/",
                     'ref2': "http://www.google.com",
                     'ref3': "http://www.google.com"}

        for key, value in refs_dict.items():
            link = tk.Label(self.frame_info_refs, text=key, fg="black", cursor="hand2")
            link.grid()
            link.bind("<Button-1>", lambda e: webbrowser.open_new(value))

        # frame right
        self.frame_run = tk.Frame(self.master)
        self.frame_run.grid(row=1, column=0, sticky=tk.W)

        tk.Label(self.frame_run, text='Required arguments:').grid()
        tk.Button(self.frame_run, text="Set surface file:", command=self.set_surf_fname).grid(row=0, column=0)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 's']['strvar'].values[0]).grid(row=0, column=1)
        tk.Button(self.frame_run, text="Set data file:", command=self.set_data_fname).grid(row=1, column=0)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'd']['strvar'].values[0]).grid(row=1, column=1)
        tk.Button(self.frame_run, text="Set output name:", command=self.set_output_fname).grid(row=2, column=0)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'o']['strvar'].values[0]).grid(row=2, column=1)

        tk.Label(self.frame_info_refs, text='Optional arguments:').grid(row=3, column=0)
        tk.Button(self.frame_run, text="Set mask file:", command=self.set_mask_fname).grid(row=4, column=0)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'm']['strvar'].values[0]).grid(row=4, column=1)
        tk.Button(self.frame_run, text="Set cluster file:", command=self.set_cluster_fname).grid(row=5, column=0)
        tk.Label(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'c']['strvar'].values[0]).grid(row=5, column=1)

        tk.Label(self.frame_run, text='Jobs:').grid(row=6, column=0)
        tk.Entry(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'j']['strvar'].values[0]).grid(row=6, column=1)

        tk.Label(self.frame_run, text='Full brain analysis:').grid(row=7, column=0)
        tk.Button(self.frame_run, textvariable=self.df_args.loc[self.df_args.flag == 'fb']['strvar'].values[0], command=self.toggle_fb).grid(row=7, column=1)

        tk.Label(self.frame_run, text='Normalization:').grid(row=8, column=0)
        tk.OptionMenu(self.frame_run, self.df_args.loc[self.df_args.flag == 'n']['strvar'].values[0], "geig", "unnorm", "rw", "sym").grid(row=8, column=1)
        tk.Button(self.frame_run, text='Run this!', command=self.run_analysis).grid(row=9, column=0)


        # frame left
        self.frame_view = tk.Frame(self.master)
        self.frame_view.grid(row=1, column=1, sticky=tk.W)
        tk.Button(self.frame_view, text='View this!', command=self.viewer).grid()


    def run_analysis(self):
        exclude_list = ['None', 'No', 'default']
        self.vb_cmd = 'vb_tool'
        for ind, arg in self.df_args.iterrows():
            val = self.df_args.loc[self.df_args.flag == arg.flag]['strvar'].values[0].get()
            if val not in exclude_list:
                if arg.flag == 'fb':
                    self.vb_cmd = f'{self.vb_cmd} -{arg.flag}'
                else:
                    self.vb_cmd = f'{self.vb_cmd} -{arg.flag} {val}'


        print(f'run the following command:{self.vb_cmd}')


    def viewer(self):
        print(f'open wb_view')


    def set_surf_fname(self):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.df_args.loc[self.df_args.flag == 's']['strvar'].values[0].set(fname)

    def set_data_fname(self):
        fname = filedialog.askopenfilename(initialdir=self.pwd, title="Select a File")
        if fname:
            self.df_args.loc[self.df_args.flag == 'd']['strvar'].values[0].set(fname)

    def set_output_fname(self):
        fname = filedialog.asksaveasfilename(initialdir=self.pwd, title="Select a File")
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



if __name__ == "__main__":
    root = tk.Tk()
    vp_toolbox_gui(root)
    root.mainloop()