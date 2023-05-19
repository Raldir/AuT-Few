
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def autolabel(rects, axis):
    for rect in rects:
        h = rect.get_height()
        ax[axis].text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.1f'%round(h, 2),
                ha='center', va='bottom')


def transpose(content):
    new_list = []
    for i in range(len(content[0])):
        new_list.append([x[i] for x in content])
    return new_list


fig, ax = fig, ax = plt.subplots(3,1, sharey=False, sharex=True, figsize=(5,5))
clrs = sns.color_palette()
# clrs = ["#000000", "#787878", "#678fda", "#f7baec", "#d46a68",]

titles = ["K=16", "K=32", "K=64"]
models = ["Zero-shot (T0)", "SetFit", "T-Few (T0)", "AuT-Few (Bart0)", "AuT-Few (T0)"]
x_title = "# Samples"
y_title = "Score"
score_file = "results_samples.txt"
score_var = "results_samples_var.txt"
fig_title="scores_over_samples_bar.png"


scores = []
with open(score_file, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split(" ")
        print(cont)
        cont = [float(x)* 100 for i, x in enumerate(cont)]
        scores.append(cont)


scores = transpose(scores) # + [mean]
print(scores)

ind = np.array([0, 1, 2, 3, 4]) #]np.arange(len(datasets))  # the x locations for the groups
width = 0.5


var = []
with open(score_var, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split(" ")
        cont = [float(x) *100  for i, x in enumerate(cont)]
        var.append(cont)
var = transpose(var)

with sns.axes_style("darkgrid"):
    ax[0].ticklabel_format(style='plain', axis='both')

    ax[0].set(ylabel=titles[0])
    ax[1].set(ylabel=titles[1])
    ax[2].set(ylabel=titles[2])
    # ax[1].set(xlabel=x_title)


    ax[0].set(ylim=(40, 80))
    ax[0].set_xticks(ind+(width))
    ax[0].set_xticklabels(models)

    ax[1].set(ylim=(40, 80))
    ax[1].set_xticks(ind+(width))
    ax[1].set_xticklabels(models)

    ax[2].set(ylim=(40, 80))
    ax[2].set_xticks(ind+(width))
    ax[2].set_xticklabels(models, fontdict={'size'   : 8, "rotation": 45})

    # ax[0].yaxis.grid(True)
    # ax[1].yaxis.grid(True)
    # ax[2].yaxis.grid(True)


    for i in range(3):#range(len(scores_list)):
        print("HEREE")
        means = scores[i]
        stds = var[i]

        for j in range(len(models)):
            rects = ax[i].bar(ind[j]+(width), means[j], width, align="center", edgecolor = "Black", alpha=1, yerr=stds[j], label=models[j],color=clrs[j])
            autolabel(rects, i)

            # ax.fill_between(titles_c, meanst-sdt, meanst+sdt ,alpha=0.2, facecolor=clrs[j])
    
    # meanst = np.array(means[len(scores)-1], dtype=np.float64)
    # ax[0].plot(titles_c, meanst, marker="s", markersize=12, linewidth=6, label="mean", c=clrs[7])

    # ax[0].legend()

    # plt.legend(bbox_to_anchor=(1.02, 1.00), loc='upper left', borderaxespad=0, prop={'size': 8})

fig.savefig(fig_title, dpi=500, bbox_inches = "tight") 
