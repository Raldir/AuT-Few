from autogluon.multimodal import MultiModalPredictor
import os
import argparse
from Config import Config
import numpy as np
import pandas as pd
import shutil
import random
from datasets import load_dataset, load_from_disk
import json
import sys

from sklearn.metrics import matthews_corrcoef


# RAFT_DATASETS = ['ade_corpus_v2', 'banking_77', 'terms_of_service', 'tai_safety_research', 'neurips_impact_statement_risks', 'overruling', 'systematic_review_inclusion',
#  'one_stop_english', 'tweet_eval_hate', 'twitter_complaints', 'semiconductor_org_types']

os.makedirs("data_cache", exist_ok=True)

label = 'label'
# backbone = 't5-small'
backbone = 'bigscience/T0_3B'


def get_dataset_reader(config):
    dataset_class = {
        "rte": RTEReader,
        "wic": WiCReader,
        "anli-r1": ANLIR1Reader,
        "anli-r2": ANLIR2Reader,
        "anli-r3": ANLIR3Reader,
        "cb": CBReader,
        "wsc": WSCFixedReader,
        "emotion": EmotionReader,
        "sst5": SST5Reader,
        "sentevalcr": SentEvalCRReader,
        "enron_spam": EnronSpamReader,
        "amazon_counterfactual_en": AmazonCounterFactReader, 
    }[config.dataset]
    return dataset_class(config)


DATASETS_OFFLINE = "/fruitbasket/datasets/datasets_offline"
MAX_EXAMPLES_PER_DATASET = 500_000

class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config, dataset_stash):
        """
        :param config:
        """
        self.config = config
        self.dataset_stash = dataset_stash

        self.templates = []
        self.train_template = self.get_template(self.config.train_template_idx)
        self.eval_template = self.get_template(self.config.eval_template_idx)

    def get_template(self, template_idx):
        return []

    def get_train_template(self):
        return self.train_template

    def get_eval_template(self):
        return self.eval_template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            orig_data = load_dataset(*self.dataset_stash, split=split, cache_dir=os.environ["HF_HOME"])
        return orig_data

    def read_few_shot_dataset(self):
        file_dir = os.path.join("data", "few_shot", self.config.dataset, f"{self.config.num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl")

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))

            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            with open(file_path, "w+") as fout:
                for example in selected_data:
                    if "enron_spam" in self.dataset_stash[0]:
                        example = {"text" : example["subject"] + ". " + example["text"], 'label' : example['label'], 'label_text' : example['label_text']}
                    fout.write(json.dumps(example) + "\n")
            return selected_data

    def _sample_few_shot_data(self, orig_data):
        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        orig_data = [x for x in orig_data]
        np.random.shuffle(orig_data)
        selected_data = orig_data[: self.config.num_shot]
        np.random.set_state(saved_random_state)
        return selected_data

    def compute_metric(self, accumulated):
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}

class ANLIR1Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli", ))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r1")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR2Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli", ))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r2")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR3Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli", ))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r3")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class WSCFixedReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "wsc.fixed"))


class RTEReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "rte"))


class WiCReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "wic"))

class SetFitReader(BaseDatasetReader):

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if split == "validation":
            split = "test"
        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            orig_data = load_dataset(*self.dataset_stash, split=split, cache_dir=os.environ["HF_HOME"])
        return orig_data

class CBReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "cb"))
        
class EmotionReader(SetFitReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("SetFit/emotion", ""))

class SST5Reader(SetFitReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("SetFit/sst5", ""))

class SentEvalCRReader(SetFitReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("SetFit/SentEval-CR", ""))

class EnronSpamReader(SetFitReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("SetFit/enron_spam", ""))

class AmazonCounterFactReader(SetFitReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("SetFit/amazon_counterfactual_en", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='dataset from RAFT benchmark', default='rte')
    parser.add_argument('--efficient_finetune', type=str, default='lora')
    parser.add_argument('--num_shot', type=int, default=16)
    parser.add_argument('--num_templates', type=int, default=30)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--few_shot_seed', default=100, type=int)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--current_ensemble', type=int, default=0)
    parser.add_argument('--no_validation', help='override save', action='store_true')

    parser.add_argument('--template', type=int, default=10)
    parser.add_argument('--majority_voting', help='override save', action='store_true')


    parser.add_argument('--run_corrupted', help='override save', action='store_true')
    parser.add_argument('--no_samples', help='override save', action='store_true')
    parser.add_argument('--run_random_replacement', help='override save', action='store_true')
    parser.add_argument('--unsupervised', help='override save', action='store_true')

    parser.add_argument('--run_corrupted_template', help='override save', action='store_true')
    parser.add_argument('--randomized_templates', help='override save', action='store_true')
    parser.add_argument('--test_qa', help='override save', action='store_true')
    parser.add_argument('--run_no_template_basic', help='override save', action='store_true')
    parser.add_argument('--run_no_template_adjusted', help='override save', action='store_true')
    parser.add_argument('--retrieve_templates', help='override save', action='store_true')

    parser.add_argument('--no_choices', help='override save', action='store_true')

    parser.add_argument('--paper_comparison', help='override save', action='store_true')
    args = parser.parse_args()

    print("Dataset {}".format(args.dataset))
    print("Seed: {}".format(args.seed))
    print("Sample Seed: {}".format(args.few_shot_seed))

    random.seed(args.seed)
    np.random.seed(args.seed)


    config = Config(kwargs={'dataset': args.dataset, 'few_shot': True, 'num_shot' : args.num_shot, "few_shot_random_seed": args.few_shot_seed})
    dataset = get_dataset_reader(config)

    train_data = dataset.read_few_shot_dataset()

    stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
    punctuation = ['.', '!', "?", ">", "<"]


    val_data = dataset.read_orig_dataset('validation')


    try:
        val_data = val_data.shuffle(seed=args.seed)
    except:
        random.shuffle(val_data)

    if args.paper_comparison:
        train_data = pd.DataFrame(train_data)
        val_data = pd.DataFrame(val_data)

        print("Training size", len(train_data))
        print("Validation size 1", len(val_data))
    else:
        val_data_1 = val_data[:args.num_shot]
        val_data_2 = val_data[args.num_shot:]

        train_data = pd.DataFrame(train_data)
        val_data_1 = pd.DataFrame(val_data_1)
        val_data_2 = pd.DataFrame(val_data_2)
        print("Training size", len(train_data))
        print("Validation size 1", len(val_data_1))
        print("Validation size 2", len(val_data_2))

    # save_path = f'{args.experiment_name}_{args.dataset}_{args.seed}_{args.few_shot_seed}_{args.num_shot}_{args.efficient_finetune}'

    if args.current_ensemble > 0 and "ensemble" in args.efficient_finetune:
        save_path = f'{args.experiment_name}_{args.dataset}_{args.seed}_{args.few_shot_seed}_{args.num_shot}_{args.efficient_finetune}_ensemble_p{str(args.current_ensemble)}'
    else:
        save_path = f'{args.experiment_name}_{args.dataset}_{args.seed}_{args.few_shot_seed}_{args.num_shot}_{args.efficient_finetune}'
    save_path = os.path.join("experiments", args.experiment_name, save_path)

    print("save_path", save_path)

    preset_templates = list(dataset.dataset_stash)
    
    if args.dataset in ["anli-r1", "anli-r2", "anli-r3"]:
        preset_templates.append('')

    custom_templates = []
    if args.run_no_template_basic:
        if args.dataset == 'rte':
            preset_templates = ["hi", "hello"]
            custom_templates = {1: {"template" : "{{premise}} {{hypothesis}}. Yes or no? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
        elif args.dataset == 'wic':
            preset_templates = ["hi", "hello"]
            custom_templates = {1: {"template" : "{{sentence1}} {{sentence2}} {{word}}. Yes or no? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
        elif args.dataset == 'wsc':
            preset_templates = ["hi", "hello"]
            custom_templates = {1: {"template" : "{{text}} {{span1_text}} {{span2_text}}. Yes or no? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
    if args.run_no_template_adjusted:
        if args.dataset == 'rte':
            preset_templates = ["hi", "hello"]
            custom_templates = {1: {"template" : "{{premise}} {{hypothesis}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
        elif args.dataset == 'wic':
            preset_templates = ["hi", "hello"]
            custom_templates = {1: {"template" : "{{sentence1}} {{sentence2}} {{word}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
        elif args.dataset == 'wsc':
            preset_templates = ["hi", "hello"]
            custom_templates = {1: {"template" : "{{text}} {{span1_text}} {{span2_text}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
    
    if args.run_corrupted_template:
        if args.dataset == 'rte':
            preset_templates = ["hi", "hello"]
            custom_templates = {1:
                                    {"template" : "Him is dog {{premise}} laugh house can {{hypothesis}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"},
                                2:
                                    {"template" : "{{premise}} campus Scott's transformed {{hypothesis}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"},
                                3:
                                    {"template" : "{{premise}} when Sears storm {{hypothesis}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}

    if args.run_corrupted_template and args.test_qa:
        if args.dataset == 'rte':
            preset_templates = ["hi", "hello"]
            custom_templates = {1:
                                    {"template" : "Answer step by step: {{premise}} \n\n {{hypothesis}}. Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"},
                                2:
                                    {"template" : "Question: {{premise}}\n\n Context: {{hypothesis}}. \n\n Yes? No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"},
                                3:
                                    {"template" : "{{premise}} \n\n {{hypothesis}}. \n\n Pick the best answer from the following options:\n\n - Yes? \n - No? ||| {{answer_choices[label]}}",
                                    "answer_choices": "Yes ||| No"}}
                                

    num_templates = 30
    if args.no_choices:
        num_templates = 255
    elif args.no_samples:
        num_templates = 128


    predictor = MultiModalPredictor.load(save_path, current_iteration = int(args.current_ensemble)) 

    setattr(predictor._model, "majority_voting", args.majority_voting)
    setattr(predictor._config.env, "per_gpu_batch_size", 1)
    # setattr(predictor._model, "total_ensembles", 5) # had to be set for a few models where it wasnt part of their attribute

    if args.paper_comparison:
        y_pred, y_pred_probs = predictor.predict(data=val_data, seed=args.seed)
        if not all([x == -1 for x in val_data[label]]):
            scores_mcc = matthews_corrcoef(val_data[label].to_numpy(), y_pred)
            scores = {'acc': (y_pred == val_data[label]).to_numpy().mean(), "mcc": scores_mcc}
            print(scores)
            with open(os.path.join(predictor.path, f'result_dev_mcc_inference_{args.majority_voting}_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
                f_out.write("{}".format(str(scores['mcc'])))
            with open(os.path.join(predictor.path, f'result_dev_inference_{args.majority_voting}_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
                f_out.write("{}".format(str(scores['acc'])))
        y_pred.to_csv(os.path.join(predictor.path, f'prediction_dev_inference_{args.majority_voting}_{args.dataset}_{str(args.seed)}.csv'))
    else:
        y_pred, y_pred_probs = predictor.predict(data=val_data_2, seed=args.seed)
        if not all([x == -1 for x in val_data_2[label]]):
            scores = {'acc': (y_pred == val_data_2[label]).to_numpy().mean()}
            print(scores)
            with open(os.path.join(predictor.path, f'result_dev_inference_{args.majority_voting}_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
                f_out.write("{}".format(str(scores['acc'])))
        y_pred.to_csv(os.path.join(predictor.path, f'prediction_dev_inference_{args.majority_voting}_{args.dataset}_{str(args.seed)}.csv'))

    # test_data= dataset.read_orig_dataset('test')
    # test_data = pd.DataFrame(test_data)

    # if not test_data.empty:
    #     y_pred = predictor.predict(data=test_data)
    #     if not all([x == -1 for x in test_data[label]]):
    #         scores = {'acc': (y_pred == test_data[label]).to_numpy().mean()}
    #         print(scores)
    #         with open(os.path.join(predictor.path, f'result_test_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
    #             f_out.write("{}".format(str(scores['acc'])))
    #     y_pred.to_csv(os.path.join(predictor.path, f'prediction_test_{args.dataset}_{str(args.seed)}.csv'))
