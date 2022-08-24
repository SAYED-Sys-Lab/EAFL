import wandb
import matplotlib
import numpy as np
from matplotlib import pylab as plt
from cycler import cycler
from pathlib import Path
import math
import sys
import os

barwidth = 0.28
barspace = 0.03
plt.style.use('seaborn-pastel')
SCATTER_SIZE = 50
FONT_SIZE = 30
fig_width_pt = 800  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
fig_width = fig_width_pt*inches_per_pt  # width in inches
golden_mean = (math.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'pdf', #'ps',
             'axes.labelsize': FONT_SIZE,
             'font.size': FONT_SIZE,
             'legend.fontsize': FONT_SIZE,
             'legend.loc': 'best', #'upper left',
             'xtick.labelsize': FONT_SIZE,
             'ytick.labelsize': FONT_SIZE,
              'lines.linewidth': 4,
              'figure.figsize': fig_size
            }
plt.rcParams.update(params)

main_dir='/homes/ahmed/amna/iFedScale/'
api = wandb.Api()
proj_names = ['energy_eff']
sample_methods = ['oort']  #can add more
grad_policy = ['YoGi']
max_epoch = 1000


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if height > 0.1:
            continue
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                '%0.2f' % height,
                ha='center', va='bottom', fontsize=14, color='red')

def smooth(y, box_pts=1):
    #return y
    box = np.ones(box_pts)/float(box_pts)
    y_smooth = np.convolve(y, box, mode='same')
    cumsum_vec = np.cumsum(np.insert(y, 0, 0))
    #y_smooth = (cumsum_vec[box_pts:] - cumsum_vec[:-box_pts]) / float(box_pts)
    #print(len(y), len(cumsum_vec), len(y_smooth))
    return y_smooth

def plot_energyeff_exps(project):
    results_dir = os.path.join(main_dir,project)
    runs = api.runs('flsys_qmul/' + project)
    t_runs = {}
    numb=1

    for run in runs:

        print(run, run.name, run.tags) #, run.config)

        if (numb == 11) or (numb == 12) or (numb==1):
            t_runs[numb] = []
            t_runs[numb].append(run)
            #print(t_runs)
        numb = numb+1

    test2 = []
    clock2 = []
    test1 = []
    clock1 = []
    test3 = []
    clock3 = []
    temp_dir = os.path.join(results_dir, 'new')
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    fig1, ax1 = plt.subplots()
    print(t_runs)

    for i, key in enumerate(t_runs):

        if (key==11):

            for run in t_runs[key]:
                run_hist1= run.scan_history(
                    keys=['Round/epoch', 'Round/clock', 'Fairness/jain_success', 'Test/acc_top_5', 'Train/loss','Round/duration',
                          'Round/dropout_battery'])
                for i, row in enumerate(run_hist1):
                    if(int(row['Round/clock'])==165600) or (int(row['Round/clock'])>165600):
                        break
                    else:

                        clock1.append((row['Round/clock']/3600))
                        test1.append(row['Round/duration'])
        if (key == 12):

            for run in t_runs[key]:
                run_hist2 = run.scan_history(keys=['Round/epoch', 'Round/clock','Fairness/jain_success' ,'Test/acc_top_5','Train/loss', 'Round/duration','Round/dropout_battery'])
                for i, row in enumerate(run_hist2):
                    if(int(row['Round/clock'])==165600)or (int(row['Round/clock'])>165600):
                        break
                    else:
                        clock2.append((row['Round/clock']/3600))
                        test2.append(row['Round/duration'])
        if (key == 1):

            for run in t_runs[key]:
                run_hist1 = run.scan_history(
                    keys=['Round/epoch', 'Round/clock', 'Fairness/jain_success', 'Round/duration','Test/acc_top_5', 'Train/loss',
                          'Round/dropout_battery'])
                for i, row in enumerate(run_hist1):
                    if (int(row['Round/clock']) == 165600) or (int(row['Round/clock']) > 165600):
                        break
                    else:

                        clock3.append((row['Round/clock'] / 3600))
                        test3.append(row['Round/duration'])

#   print(len(test1))
#  print(len(clock1))

    a=smooth(test1, 3)
    b=smooth(test2, 3)
    c=smooth(test3, 3)
    ax1.plot(clock1, a,label='EAFL', color='r')
    ax1.plot(clock2,b,label='Oort', color='b' )
    ax1.plot(clock3, c, label='Random', color='g')
    ax1.set_ylabel('Round Duration', fontsize=FONT_SIZE)
    ax1.set_xlabel('Training Runtime (Hours)', fontsize=FONT_SIZE)
    ax1.legend(title='Methods', ncol=2)
    ax1.grid(True)
    fig1.set_tight_layout(True)
    fig1.savefig(temp_dir + "/round13.png", bbox_inches='tight')

    #### Legend
    figLegend = plt.figure(figsize=(2, 0.1))


print(len(sys.argv))
if sys.argv[1] == 'energy_eff':
    for project in proj_names:
        plot_energyeff_exps(project)