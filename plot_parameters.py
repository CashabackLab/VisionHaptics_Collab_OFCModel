"""
Plot parameters for running model simulations. Imported by the simulation scripts.
"""

import os

plots_directory = os.getcwd() + r'\\Plots\\' # directory where the plots will be saved

figure_theme_color = 'w' # 'k': dark theme figures, 'w': light theme figures

font_weight = 'bold' # font weight control
axis_fsize = 36 # axis label font size
legend_fsize = 28 # legend text font size
tick_fsize = 34 # tick label font size
pval_fsize = 30 # p value text font size
axis_linewidth = 3 # axis line width
boxplot_linewidth = 8 # width of the lines making the boxes in the boxplot
axis_color = 'w' if figure_theme_color == 'k' else 'k' # axis lines and ticks colors
annotation_color = (0.7, 0.7, 0.7) if figure_theme_color == 'k' else 'tab:grey' # annotation elements text, arrows colors
condition_names = {'NN': 'NoVision-NoHaptic', 'H': 'NoVision-Haptic', 'V': 'Vision-NoHaptic', 'VH': 'Vision-Haptic'}
condition_names2 = {'NN': 'NoVision\nNoHaptic', 'H': 'NoVision\nHaptic', 'V': 'Vision\nNoHaptic', 'VH': 'Vision\nHaptic'}
condition_colors = {'NN': '#0BB8FD', 'H': '#23537F',  'V': '#FD8B0B', 'VH': '#C70808'} if figure_theme_color == 'w' else {'NN': '#0BB8FD', 'H': '#3F3FFF',  'V': '#FD8B0B', 'VH': '#C70808'}

condition_plotorder = [0, 2, 1, 3] # plot order for the boxes in the box plot. 0 - NN, 1 - H, 2 - V, 3 - VH
individual_data_color = (0.6, 0.6, 0.6) if figure_theme_color == 'k' else 'silver' # color for individual participant circles in the boxplot
individual_line_color = (0.4, 0.4, 0.4) if figure_theme_color == 'k' else 'tab:grey' # color for individual participant lines in the boxplot
participant_color = ('#FFC000', '#9C5BCD') if figure_theme_color == 'k' else ('#FFC000', '#9C5BCD') # left and right participant colors for animations
target_color = 'silver' if figure_theme_color == 'k' else 'plum' # color of the target for 2d trajecotry and animations

plot_params = {'font_weight': font_weight, 'axis_fsize': axis_fsize, 
                'tick_fsize': tick_fsize, 'pval_fsize': pval_fsize, 'legend_fsize': legend_fsize,
                'axis_linewidth': axis_linewidth, 'boxplot_linewidth': boxplot_linewidth,
                'axis_color': axis_color,
                'figure_theme_color': figure_theme_color,
                'annotation_color': annotation_color,
                'individual_line_color': individual_line_color
}