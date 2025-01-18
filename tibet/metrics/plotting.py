import matplotlib.pyplot as plt
import seaborn as sns
from tibet.metrics.MAD_func import get_variance
from tibet.metrics.clean_concepts import get_concepts
from skimage import io
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_CAS_data(p_dict, debug=False):

  biases = list(p_dict['result'].keys())

  CAS_data = {}
  CAS_labels = {}
  variance = []
  for bias_axis in biases:
    out = get_variance(p_dict, bias_axis, debug=debug)
    variance.append(out[0])
    # var = out[1]
    # bias_idx = list(p_dict['result'].keys()).index(bias_axis) + 1
    # var = var[bias_idx]
    #variance.append(var)

    CAS_data[bias_axis] = out[2]
    data_list = []
    for cfix, cf in enumerate(p_dict['result'][bias_axis]):
      data_list.append('CF'+str(cfix+1))
    CAS_labels[bias_axis] = data_list

  biases = [x for _, x in sorted(zip(variance, biases), reverse=True)]
  variance = sorted(variance, reverse=True)
  variance_data = {'Category': biases, 'Variance': [float(v)/sum(variance) for v in variance]}

  return variance_data, CAS_data, CAS_labels


def plot_BAV_and_CAS(variance_data, CAS_data, CAS_labels, save_dir=None):

  sorted_CAS_data = {}
  sorted_CAS_labels = {}

  # Sorting the lists according to the contents of the first list
  for category in CAS_data:
      # Pair percentages with labels and sort by percentages
      paired = sorted(zip(CAS_data[category], CAS_labels[category]), reverse=False)

      # Unzip the pairs back into separate lists
      sorted_CAS, sorted_CAS_labels_indidivual = zip(*paired)

      # Update the dictionaries with the sorted lists
      sorted_CAS_data[category] = sorted_CAS
      sorted_CAS_labels[category] = sorted_CAS_labels_indidivual
  CAS_data=sorted_CAS_data
  CAS_labels=sorted_CAS_labels

  # Set the aesthetic style of the plots
  sns.set_style("whitegrid")

  num_rows = max(len(variance_data['Category']), 2)

  # Create a figure and a grid of subplots
  fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5/3*num_rows), gridspec_kw={'width_ratios': [1, 3]})

  # Color for variance bars
  variance_color = 'salmon'

  # Color for percentage histograms
  hist_color = 'green'

  # Normalize variance data for the bar chart
  max_variance = max(variance_data['Variance'])
  normalized_variance = [x / max_variance for x in variance_data['Variance']]

  # Plotting the percentage data as histograms in the first column
  for i, category in enumerate(variance_data['Category']):
      labels = CAS_labels[category]
      values = CAS_data[category]
      colors = sns.light_palette(hist_color, n_colors=len(values)+1)[1:]  # Use lighter shades for each category
      axs[i, 0].bar(labels, values, color=colors)
      #axs[i, 0].set_title(category)
      axs[i, 0].set_xticks(axs[i, 0].get_xticks())
      axs[i, 0].set_xticklabels(labels, fontsize=8) #rotation=45
      axs[i, 0].set_ylim(0, 1)  # Normalized scale
      axs[i, 0].set_yticks(axs[i, 0].get_yticks())
      axs[i, 0].set_yticklabels(axs[i,0].get_yticks(), fontsize=8)
      axs[i, 0].grid(False)
      axs[0, 0].set_title("CAS")

  # Plotting the variance as horizontal bars in the second column
  for i, (category, var) in enumerate(zip(variance_data['Category'], normalized_variance)):
      colors = sns.light_palette(variance_color, n_colors=len(variance_data['Category'])+1)[len(variance_data['Category'])-i:]
      axs[i, 1].barh(category, var, color=colors)
      axs[i, 1].set_xlim(0, 1)  # Normalized scale
      axs[i, 1].set_xlabel(' '.join(category.split('_')),fontsize=10)
      axs[i, 1].set_yticklabels([])
      axs[i, 1].tick_params(left=False, bottom=False)  # Hide ticks
      axs[i, 1].set_xticks([])  # Hide x-tick labels
      axs[i, 1].grid(False)
      axs[0, 1].set_title("MAD")

  #plt.subplots_adjust(wspace=0, hspace=0)

  # Adjust the layout
  plt.tight_layout()
  #plt.gcf().set_size_inches(10/2, 5/3*num_rows/2)

  if save_dir is not None:
    print("Saving plot")
    plt.savefig(save_dir)
  
  # Show the plot
  plt.show()
