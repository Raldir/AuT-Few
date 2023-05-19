import os
import argparse
import pathlib
from glob import glob
import yaml
import statistics


def draw_plot(data_all, names, title):
    import matplotlib.pyplot as plot
    import numpy as np
    #Generate the dat
    plot.rcParams['font.size'] = '6'
    plot.boxplot(data_all)

    plot.xticks([x for x in range(1, len(names) + 1)], names)

    plot.title(title)


    plot.savefig(fname=f"{title}.png")

DATASETS = ["rte", "wsc", "wic", "anli-r1", "anli-r2", "anli-r3", "cb", "emotion", "enron_spam", "amazon_counterfactual_en", "sentevalcr", "sst5"]

VAL_RESULTS_FILE = "validation_score.txt"


def collect_results(experiment_name, dataset, pet, inference=False, score="acc"):
    print(experiment_name, dataset)
    f_out = open(os.path.join("experiments", experiment_name, f"results_{experiment_name}.txt"), "a")
    file_path = pathlib.Path(__file__).parent.resolve()
    all_paths  = glob(os.path.join(str(file_path), "experiments", experiment_name + "/*"))
    all_results_val = []
    all_results_val_2 =[]
    all_results_val_epoch = []
    for path_or in all_paths:
        path = path_or.split('/')[-1]
        if path.endswith(".txt") or path.endswith(".sh"):
            continue
        path = path.split(experiment_name)[1][1:]
        dataset_name = dataset
        pet_name = path.split('_')[-1]
        if "ensemble" in pet_name or 'p' in pet_name:
            pet_name = "ia3"
        if pet_name not in pet:
            continue
        seed = path.split("_")[1]
        sample_seed = path.split("_")[2]
        num_samples = path.split("_")[-2]
        all_files = glob(path_or + "/*", recursive = True)
        if not inference:
            if score == 'acc' or (score == "combined" and dataset_name != "amazon_counterfactual_en"):
                VAL_RESULTS_2_FILE= f"result_dev_{dataset_name}_"
            elif score == "mcc" or (score == "combined" and dataset_name == "amazon_counterfactual_en"):
                VAL_RESULTS_2_FILE= f"result_dev_mcc_{dataset_name}_"
        else:
            if score == "acc":
                VAL_RESULTS_2_FILE= f"result_dev_inference_True_{dataset_name}_"
            elif score == "mcc":
                VAL_RESULTS_2_FILE= f"result_dev_mcc_inference_True_{dataset_name}_"
        val_results = None
        val_results_epoch = None
        val_results_2 = None
        for file in all_files:
            if VAL_RESULTS_FILE in file:
                with open(file, "r") as f:
                    content = yaml.safe_load(f)
                    for key, item in content.items():
                        assert len(content.items()) == 1
                        val_results_epoch = key
                        val_results = item
                        all_results_val.append(val_results)
                        all_results_val_epoch.append(val_results_epoch)
            if VAL_RESULTS_2_FILE in file:
                with open(file, "r") as f:
                    content = f.readlines()
                    for line in content:
                        val_results_2 = float(line.strip())
                        all_results_val_2.append(val_results_2)

        res_to_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(experiment_name, dataset_name, pet_name, seed, sample_seed, num_samples, val_results, val_results_epoch, val_results_2)
        f_out.write(res_to_string + '\n')

    if len(all_results_val_2) < 2:
        print("Only one datapoint. Not computing average metrics.")
        return [None, None, None]
    variance = statistics.stdev(all_results_val_2)
    print(all_results_val_2)
    mean = statistics.mean(all_results_val_2)
    min_value = min(all_results_val_2)
    max_value = max(all_results_val_2)
    if experiment_name == 'ablation_zero_shot_':
        mean_epochs = "NA"
    else:
        try:
            mean_epochs = statistics.mean(all_results_val_epoch)
        except:
            mean_epochs = "NA"
    # print("Mean: {}, Variance:: {}, Min: {}, Max: {}".format(mean, variance, min_value, max_value))
    summary_string = "{}\t{}\t{}\t{}\t{}\t{}\n".format(experiment_name, dataset, mean, variance, min_value, max_value)
    all_stats = [experiment_name, dataset, mean, variance, min_value, max_value]
    return [all_results_val_2, all_stats, summary_string]

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--score', type=str, default='combined')
    parser.add_argument('--inference', help='override save', action='store_true')
    parser.add_argument('--old_layout', help='override save', action='store_true')
    parser.add_argument("--print_option", type=str, default="all")
    parser.add_argument("--model", type=str, default="T0", choices=["T0", "BART0", "Flan"])
    args = parser.parse_args()

    # for dataset in DATASETS:
    pet = ["ia3", "lora"]
    pet_text = '_'.join(pet)

    if args.model == "T0":
        DATASETS = ["rte", "wsc", "wic", "anli-r1", "anli-r2", "anli-r3", "cb", "emotion", "enron_spam", "amazon_counterfactual_en", "sentevalcr", "sst5"]
    elif args.model == "BART0":
        DATASETS = ["rte", "anli-r1", "anli-r2", "anli-r3", "cb", "emotion", "enron_spam", "amazon_counterfactual_en", "sentevalcr", "sst5"]
    elif args.model == "Flan":
        DATASETS = ["enron_spam", "amazon_counterfactual_en", "sentevalcr", "sst5"]

    ALL_EXPERIMENTS = [args.experiment_name]
    all_all_stats = []
    for exp in ALL_EXPERIMENTS:
        summary_strings = []
        for dataset in DATASETS:
            # exp = os.path.join("experiments", exp)
            results, all_stats, summary_string = collect_results(exp, dataset, pet, args.inference, args.score)
            if results == None:
                continue
            all_all_stats.append(all_stats)
            summary_strings.append(summary_string)
        
            begin_index = None
            if args.print_option == "mean":
                begin_index = 2
            elif args.print_option == "variance":
                begin_index = 3
            elif args.print_option == 'min':
                begin_index = 4
            elif args.print_option == 'max':
                begin_index = 5

            if args.inference:
                out_file = os.path.join("experiments", exp, f"summary_results_inference_{args.print_option}_{args.score}_{pet_text}.txt")
            else:
                out_file = os.path.join("experiments", exp, f"summary_results_{args.print_option}_{args.model}_{args.score}_{pet_text}.txt")
                
            with open(out_file, "w") as f_out:
                if not args.old_layout:
                    for i in range(1, len(all_all_stats[0])):
                        if begin_index and i != begin_index and i != 1:
                            continue
                        content = []
                        if i > 1 and i < 4:
                            overall_mean = str(sum([x[i] for x in all_all_stats]) / len(all_all_stats))
                            content.append(overall_mean)
                        else:
                            if i == 1:
                                content.append("Averaged over datasets")
                            else:
                                content.append("--")
                        for j in range(len(all_all_stats)):
                            content.append(str(all_all_stats[j][i]))
                            

                        header_index = {0: "experiment_name", 1: "dataset", 2: "mean", 3: "variance", 4: "min", 5: "max"}
                        f_out.write(all_all_stats[0][0] + "\t")
                        f_out.write(header_index[i] + '\t')
                        f_out.write('\t'.join(content))
                        f_out.write('\n')
                else:
                    for sum in summary_strings:
                        f_out.write(sum)


    # for dataset in DATASETS:  
    #     dataset_results = []
    #     for exp in ALL_EXPERIMENTS:
    #         results, summary_string = collect_results(exp, dataset, pet)
    #         if results == None:
    #             continue
    #         dataset_results.append(results)

    #     title = f'boxplot_replication_{dataset}_{pet_text}_' + '_'.join(ALL_EXPERIMENTS)

    #     draw_plot(dataset_results, ['_'.join(x.split("_")[1:]) for x in ALL_EXPERIMENTS], title)


