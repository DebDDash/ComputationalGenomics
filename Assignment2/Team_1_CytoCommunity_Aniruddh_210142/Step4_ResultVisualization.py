import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sci_palettes
import os
import argparse
import shutil
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42     # make text in plot editable in AI.
# print(sci_palettes.PALETTES.keys())     # used for checking all color schemes of different journals.
# sci_palettes.PALETTES["d3_category20"]     # see detailed color code
# register a specific palette for TCN coloring.
sci_palettes.register_cmap("d3_category20")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str,
                    required=False, default='merfish', help="Image_Name")
args = parser.parse_args()


# Hyperparameters
InputFolderName = "./data/"
Image_Name = args.image


# Import the cell type list used for matching color palettes across different cell type plots.
unique_cell_type_df = pd.read_csv(
    "./Step1_Output/UniqueCellTypeList.txt",
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["UniqueCellType"],  # set our own names for the columns
)
UniqueCellType_vec = unique_cell_type_df['UniqueCellType'].values.tolist()


# Import target cellular spatial graph x/y coordinates.
GraphCoord_filename = InputFolderName + Image_Name + "_Coordinates.txt"
x_y_coordinates = pd.read_csv(
    GraphCoord_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    # set our own names for the columns
    names=["x_coordinate", "y_coordinate"],
)
target_graph_map = x_y_coordinates
# target_graph_map["y_coordinate"] = 0 - target_graph_map["y_coordinate"]  # for consistent with original paper. Don't do this is also ok.


# Import cell type label.
CellType_filename = InputFolderName + Image_Name + "_CellTypeLabel.txt"
cell_type_label = pd.read_csv(
    CellType_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["cell_type"],  # set our own names for the columns
)
# Add cell type labels to target graph x/y coordinates.
target_graph_map["Cell_Type"] = cell_type_label.cell_type
# Below is for matching color palettes across different cell type plots.
target_graph_map["Cell_Type"] = pd.Categorical(
    target_graph_map["Cell_Type"], UniqueCellType_vec)


# Import the final TCN labels to target graph x/y coordinates.
LastStep_OutputFolderName = "./Step3_Output_" + Image_Name + "/"
target_graph_map["TCN_Label"] = np.loadtxt(
    LastStep_OutputFolderName + "TCNLabel_MajorityVoting.csv", dtype='int', delimiter=",")
# Converting integer list to string list for making color scheme discrete.
target_graph_map.TCN_Label = target_graph_map.TCN_Label.astype(str)


# -----------------------------------------Generate plots-------------------------------------------------#
ThisStep_OutputFolderName = "./Step4_Output_" + Image_Name + "/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)


# Plot x/y map with "TCN_Label" coloring.
TCN_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, hue="TCN_Label",
                           palette="d3_category20", alpha=1.0, s=20.0, legend="full")   # 20 colors at maximum.
# Hide all four spines
TCN_plot.spines.right.set_visible(False)
TCN_plot.spines.left.set_visible(False)
TCN_plot.spines.top.set_visible(False)
TCN_plot.spines.bottom.set_visible(False)
TCN_plot.set(xticklabels=[])  # remove the tick label.
TCN_plot.set(yticklabels=[])
TCN_plot.set(xlabel=None)  # remove the axis label.
TCN_plot.set(ylabel=None)
TCN_plot.tick_params(bottom=False, left=False)  # remove the ticks.
# Place legend outside top right corner of the CURRENT plot
TCN_plot.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
TCN_fig = TCN_plot.get_figure()
# Save the CURRENT figure.
TCN_fig_filename1 = ThisStep_OutputFolderName + "TCN_" + Image_Name + ".pdf"
TCN_fig.savefig(TCN_fig_filename1)
TCN_fig_filename2 = ThisStep_OutputFolderName + "TCN_" + Image_Name + ".png"
TCN_fig.savefig(TCN_fig_filename2)
# TCN_plot.close()


# Plot x/y map with "Cell_Type" coloring.
CellType_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, hue="Cell_Type",
                                palette=sns.color_palette("husl", 50), alpha=1.0, s=20.0, legend="brief")  # 30 colors at maximum.
# Hide all four spines
CellType_plot.spines.right.set_visible(False)
CellType_plot.spines.left.set_visible(False)
CellType_plot.spines.top.set_visible(False)
CellType_plot.spines.bottom.set_visible(False)
CellType_plot.set(xticklabels=[])  # remove the tick label.
CellType_plot.set(yticklabels=[])
CellType_plot.set(xlabel=None)  # remove the axis label.
CellType_plot.set(ylabel=None)
CellType_plot.tick_params(bottom=False, left=False)  # remove the ticks.
# Place legend outside top right corner of the CURRENT plot
CellType_plot.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
CellType_fig = CellType_plot.get_figure()
# Save the CURRENT figure.
CellType_fig_filename1 = ThisStep_OutputFolderName + \
    "CellType_" + Image_Name + ".pdf"
CellType_fig.savefig(CellType_fig_filename1)
CellType_fig_filename2 = ThisStep_OutputFolderName + \
    "CellType_" + Image_Name + ".png"
CellType_fig.savefig(CellType_fig_filename2)
# CellType_plot.close()


# Export result dataframe: "target_graph_map".
TargetGraph_dataframe_filename = ThisStep_OutputFolderName + \
    "ResultTable_" + Image_Name + ".csv"
# remove row index.
target_graph_map.to_csv(TargetGraph_dataframe_filename,
                        na_rep="NULL", index=False)
