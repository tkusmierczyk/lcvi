# -*- coding: utf-8 -*-
""" Auxiliary functions for plotting. """

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns; sns.set(); sns.set(font_scale=1.0); sns.set_style("white"); 

import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np


GREEN = "limegreen"
BLUE = "dodgerblue"
RED = "salmon"
    
BLUES = ["dodgerblue", "#F0F8FF", "#E6E6FA", "#B0E0E6", "#ADD8E6", "#87CEFA", "#87CEEB", "#00BFFF", 
         "#B0C4DE", "#1E90FF", "#6495ED", "#4682B4", "#5F9EA0", "#7B68EE", "#6A5ACD", "#483D8B", "#4169E1", 
         "#0000FF", "#0000CD", "#00008B", "#000080", "#191970", "#8A2BE2", "#4B0082"]
REDS = ["salmon", "#FFA07A","#E9967A","#F08080","#CD5C5C","#DC143C","#B22222",
        "#FF0000","#8B0000","#800000","#FF6347","#FF4500","#DB7093"]    
COLORS = ['dodgerblue', 'salmon',  'limegreen', 'teal', 'mediumspringgreen', 'violet',  'crimson']


def _reset_mpl_config(font_size = 17, cmbright=True):
    mpl.rcParams.update(mpl.rcParamsDefault) #reset to defaults
        
    SMALL_SIZE = font_size-4
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rc('font',**{'family':'serif','serif':['Times'], "weight": "normal"})
    plt.rc('text', usetex=True)
    plt.rc('mathtext', fontset='stix')  #['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
    
    mpl.rcParams['text.latex.preamble'] = [
            r'\usepackage{mathtools}',
            r'\usepackage{amsmath}',
            r'\usepackage{amsfonts}', 
            r'\usepackage{microtype}',    
            r'\usepackage{arydshln}',
    ] + ([r'\usepackage{cmbright}'] if cmbright else [])

    
def _create_fig(bottom=0.2, left=0.125, right=0.95, top=0.95):
    fig = plt.figure(figsize=(6.4, 4.8), dpi=72)
    fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top) 
    
    
def start_plotting(cmbright=True, bottom=0.2, left=0.125, right=0.95, top=0.95, font_size=17*1.5):
    _reset_mpl_config(cmbright=cmbright, font_size=font_size)
    _create_fig(bottom=bottom, left=left, right=right, top=top)
    

def running_mean(x, N=3):
    l = N//2    
    return [np.mean(x[max(i-l,0): min(i+l+1, len(x))]) for i in range(len(x))]


