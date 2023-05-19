
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def transpose(content):
    return content
    new_list = []
    for i in range(len(content[0])):
        new_list.append([x[i] for x in content])
    return new_list


fig, ax = plt.subplots(2,1, sharey=True, figsize=(12,5))
clrs = sns.color_palette("husl", 13)

clrs = ["#000000", "#848690", "#678fda", "#f7baec", "#d97978"]

titles = ["inverted", "random", "dataset", "handcrafted", "best-single"]
datasets = ["RTE", "WSC", "WiC", "ANLI-r1", "ANLI-r2", "ANLI-r3", "CB", "Emotion", "Enron", "Amazon", "CR", "SST-5", "Average"]
x_title = "Answer Choices"
y_title = "Score"
score_file = "results_ablation_choices.txt"
score_var = "results_ablation_choices_var_new.txt"
fig_title="prompt_ablation_bar_color.png"


titles_t = ["null", "random", "general", "handcrafted"]
x_title_t = "Templates"
score_file_t = "results_ablation_templates_new.txt"
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

scores_t = [scores_t[i] + [mean_t[i]] for i in range(len(scores_t))]
# scores_t = transpose(scores_t)  + [mean_t]



var_t = []
with open(score_var_t, "r") as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        cont = line.split("\t")
        cont = [float(x) * 100  for i, x in enumerate(cont)]
        var_t.append(cont)
var_t = transpose(var_t)

mean_var_t  = [(sum(x)/len(var_t[0])) for x in var_t]

var_t = [var_t[i] + [mean_var_t[i]] for i in range(len(var_t))]

print("Variance", mean_var_t)


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

scores = [scores[i] + [mean[i]] for i in range(len(scores))]

# scores = transpose(scores)  + [mean]
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

mean_var  = [(sum(x)/len(var[0])) for x in var]

print("Variance", mean_var)

var = [var[i] + [mean_var[i]] for i in range(len(var))]


ind = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]) #]np.arange(len(datasets))  # the x locations for the groups
width = [0.2, 0.15]    # the width of the bars


scores_list = [scores_t, scores]
var_list = [var_t, var]
titles_list = [titles_t, titles]

with sns.axes_style("darkgrid"):
    ax[0].ticklabel_format(style='plain', axis='both')
    ax[0].set(ylabel=y_title)
    ax[1].set(ylabel=y_title)

    # ax[0].set(xlabel=x_title_t)
    # ax[1].set(xlabel=x_title)

    ax[0].set(ylim=(30, 100))
    ax[0].set_xticks(ind+(width[0]))
    ax[0].set_xticklabels(datasets)

    ax[1].set(ylim=(0, 100))
    ax[1].set_xticks(ind+(width[1]))
    ax[1].set_xticklabels(datasets)

    ax[0].yaxis.grid(True)
    ax[1].yaxis.grid(True)


    for i in range(2):#range(len(scores_list)):
        print("HEREE")
        means = scores_list[i]
        stds = var_list[i]
        titles_c = titles_list[i]

        print("NOW", i)
        for j in range(len(titles_c)):

            print(len(means[j]), len(stds[j]))
            meanst = np.array(means[j], dtype=np.float64)
            sdt = np.array(stds[j], dtype=np.float64)           
            ax[i].bar(ind+(width[i]*j), meanst, width[i], yerr=sdt, align="center", alpha=1, edgecolor = "Black", label=datasets[j], color=clrs[j])
            # ax[i].fill_between(titles_c, meanst-sdt, meanst+sdt ,alpha=0.1, facecolor=clrs[j+1])

    # ax.legend()

    ax[0].vlines(x=12.4, ymin=0, ymax=100, colors='purple', ls=':', lw=2, label='_nolegend_')
    ax[1].vlines(x=12.4, ymin=0, ymax=100, colors='purple', ls=':', lw=2, label='_nolegend_')

    ax[0].legend(titles_t, title="Templates", bbox_to_anchor=(0.50, 1.25), loc='upper center', ncol=len(titles_t), prop={'size': 7})
    ax[1].legend(titles, title="Answer Choices", bbox_to_anchor=(0.50, 1.25), loc='upper center', ncol=len(titles), prop={'size': 7})

plt.subplots_adjust(hspace=0.5)
fig.savefig(fig_title, dpi=500) 
