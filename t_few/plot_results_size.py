
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def transpose(content):
    new_list = []
    for i in range(len(content[0])):
        new_list.append([x[i] for x in content])
    return new_list


fig, ax = plt.subplots(figsize=(5,5))
clrs = sns.color_palette("husl", 5)

titles = ["16", "32", "64"]
models = ["Zero-shot (T0)", "SetFit", "T-Few", "AuT-Few (Bart0)", "AuT-Few (T0)"]
x_title = "# Samples"
y_title = "Score"
score_file = "results_samples.txt"
score_var = "results_samples_var.txt"
fig_title="scores_over_samples.png"


scores = []
with open(score_file, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split(" ")
        print(cont)
        cont = [float(x)* 100 for i, x in enumerate(cont)]
        scores.append(cont)


# scores = transpose(scores)  + [mean]
print(scores)

var = []
with open(score_var, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split(" ")
        cont = [float(x) *100 / 4 for i, x in enumerate(cont)]
        var.append(cont)
# var = transpose(var)

with sns.axes_style("darkgrid"):
    ax.ticklabel_format(style='plain', axis='both')
    ax.set(ylabel=y_title)
    ax.set(xlabel=x_title)

    ax.set(ylim=(35, 70))

    means = scores
    stds = var
    titles_c = titles

    for j in range(len(scores)):
        meanst = np.array(means[j], dtype=np.float64)
        sdt = np.array(stds[j], dtype=np.float64)
        print(sdt)
        if j == 0:
            ax.plot(titles_c, meanst, marker="s", alpha=1, linestyle='--', label=models[j], c=clrs[j])
        else:
            ax.plot(titles_c, meanst, marker="s", alpha=1, label=models[j], c=clrs[j])
        ax.fill_between(titles_c, meanst-sdt, meanst+sdt ,alpha=0.2, facecolor=clrs[j])
    
    # meanst = np.array(means[len(scores)-1], dtype=np.float64)
    # ax[0].plot(titles_c, meanst, marker="s", markersize=12, linewidth=6, label="mean", c=clrs[7])

    ax.legend()
    # plt.legend(bbox_to_anchor=(1.02, 1.00), loc='upper left', borderaxespad=0, prop={'size': 8})

fig.savefig(fig_title) 
