
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def transpose(content):
    new_list = []
    for i in range(len(content[0])):
        new_list.append([x[i] for x in content])
    return new_list


fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,5))
clrs = sns.color_palette("husl", 13)

titles = ["inverted", "random", "dataset", "handcrafted", "best-single"]
datasets = ["RTE", "WSC", "WiC", "ANLI-r1", "ANLI-r2", "ANLI-r3", "CB", "Emotion", "Enron", "Amazon-CF", "CR", "SST-5"]
x_title = "Answer Choices"
y_title = "Score"
score_file = "results_ablation_choices.txt"
score_var = "results_ablation_choices_var_new.txt"
fig_title="prompt_ablation.png"


titles_t = ["null", "random", "general", "handcrafted"]
x_title_t = "Templates"
score_file_t = "results_ablation_templates.txt"
score_var_t = "results_ablation_templates_var_new.txt"



scores_t = []

with open(score_file_t, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split("\t")
        print(cont)
        cont = [float(x)* 100 for i, x in enumerate(cont)]
        scores_t.append(cont)

mean_t = [(sum(x)/len(scores_t[0])) for x in scores_t]

print("mean", mean_t)

scores_t = transpose(scores_t)  + [mean_t]


var_t = []
with open(score_var_t, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split("\t")
        cont = [float(x) * 100  for i, x in enumerate(cont)]
        var_t.append(cont)
var_t = transpose(var_t)


scores = []
with open(score_file, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split("\t")
        print(cont)
        cont = [float(x)* 100 for i, x in enumerate(cont)]
        scores.append(cont)

mean = [(sum(x)/len(scores[0])) for x in scores]

print("mean", mean)

scores = transpose(scores)  + [mean]
print(scores)

var = []
with open(score_var, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split("\t")
        cont = [float(x) *100  for i, x in enumerate(cont)]
        var.append(cont)
var = transpose(var)


scores_list = [scores_t, scores]
var_list = [var_t, var]
titles_list = [titles_t, titles]

with sns.axes_style("darkgrid"):
    ax[0].ticklabel_format(style='plain', axis='both')
    ax[0].set(ylabel=y_title)
    ax[0].set(xlabel=x_title_t)
    ax[1].set(xlabel=x_title)

    ax[0].set(ylim=(40, 100))

    for i in range(len(scores_list)):
        means = scores_list[i]
        stds = var_list[i]
        titles_c = titles_list[i]

        print("NOW", i)
        for j in range(len(scores)-1):
            meanst = np.array(means[j], dtype=np.float64)
            sdt = np.array(stds[j], dtype=np.float64)
            ax[i].plot(titles_c, meanst, marker="s", alpha=0.4, label=datasets[j], c=clrs[j+1])
            ax[i].fill_between(titles_c, meanst-sdt, meanst+sdt ,alpha=0.1, facecolor=clrs[j+1])
        
        meanst = np.array(means[len(scores)-1], dtype=np.float64)
        ax[i].plot(titles_c, meanst, marker="s", markersize=12, linewidth=6, label="mean", c=clrs[7])

    # ax.legend()
    plt.legend(bbox_to_anchor=(1.02, 1.00), loc='upper left', borderaxespad=0, prop={'size': 8})

fig.savefig(fig_title) 
