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
import copy
from sklearn.metrics import matthews_corrcoef

os.makedirs("data_cache", exist_ok=True)

label = 'label'
backbone = 'bigscience/T0_3B'

def get_dataset_reader(config):
    dataset_class = {
        "rte": RTEReader,
        "wic": WiCReader,
        "anli-r1": ANLIR1Reader,
        "anli-r2": ANLIR2Reader,
        "anli-r3": ANLIR3Reader,
        "wsc": WSCFixedReader,
        "cb": CBReader,
        "emotion": EmotionReader,
        "sst5": SST5Reader,
        "sentevalcr": SentEvalCRReader,
        "enron_spam": EnronSpamReader,
        "amazon_counterfactual_en": AmazonCounterFactReader, 
        "ade_corpus_v2": RaftReader,
        "banking_77": RaftReader,
        "terms_of_service": RaftReader,
        "tai_safety_research": RaftReader,
        "neurips_impact_statement_risks": RaftReader,
        "overruling": RaftReader,
        "systematic_review_inclusion": RaftReader,
        "one_stop_english": RaftReader,
        "tweet_eval_hate": RaftReader,
        "twitter_complaints": RaftReader,
        "semiconductor_org_types": RaftReader,
    }[config.dataset]
    return dataset_class(config)


DATASETS_OFFLINE = "/fruitbasket/datasets/datasets_offline"

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
        self.orig_data = None

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
            orig_data = load_dataset(*self.dataset_stash, split=split, cache_dir="cache/")
        self.orig_data = orig_data
        return orig_data

    def read_few_shot_dataset(self):
        file_dir = os.path.join("data", "few_shot", self.config.dataset, f"{self.config.num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{self.config.few_shot_random_seed}_{self.config.balanced_sampling}_seed.jsonl")

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
        if self.config.balanced_sampling:
            orig_data = [x for x in orig_data]
            num_classes = len(set([x['label'] for x in orig_data]))
            orig_data = pd.DataFrame(orig_data)
            sampled_data = []
            for cl in range(num_classes):
                subset = orig_data.query(f"label == {cl}")
                if len(subset) > self.config.num_shot:
                    sampled_data.append(subset.sample(self.config.num_shot, random_state=self.config.few_shot_random_seed, replace=False))
                else:
                    sampled_data.append(subset)
            sampled_data = pd.concat(sampled_data)
            sampled_data = list(sampled_data.T.to_dict().values())
            counts = {x: 0 for x in range(num_classes)}
            for entry in sampled_data:
                counts[entry['label']] +=1
            return sampled_data
        else:
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
        if split == "validation": # DO NOT USE TEST SET FOR NOW
            split = "dev"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r1")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR2Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli", ))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "dev"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r2")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR3Reader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("anli", ))

    def read_orig_dataset(self, split):
        if split == "validation":
            split = "dev"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r3")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data

class CBReader(BaseDatasetReader):
    def __init__(self, config):
        super().__init__(config, dataset_stash=("super_glue", "cb"))

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
        if split == "validation" and self.config.dataset in ["amazon_counterfactual_en", "sentevalcr", "enron_spam"]:
            split="train"
        # if split == "validation":
        #     split = "test"
        if os.path.exists(DATASETS_OFFLINE):
            orig_data = load_from_disk(os.path.join(DATASETS_OFFLINE, *self.dataset_stash))[split]
        else:
            orig_data = load_dataset(*self.dataset_stash, split=split, cache_dir="cache/")
        return orig_data

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


class RaftReader(object):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset
        self.orig_data = load_dataset("ought/raft", name=self.dataset_name)
        self.answer_choices = self.orig_data["train"].features["Label"].names[1:]
        if self.config.dataset == "banking_77" and config.cleaned_answer_choices_b77:
            self.answer_choices = [answer.replace("_", " ").replace(". ", " ") for answer in self.answer_choices]
        self.dataset_stash = ("raft", self.dataset_name)

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if self.config.raft_cross_validation:
            orig_data = [example for example in self.orig_data["train"]]
            if split == "train":
                orig_data = (
                    orig_data[: self.config.raft_validation_start] + orig_data[self.config.raft_validation_start + 10 :]
                )
                assert len(orig_data) == 40
            elif split == "validation":
                orig_data = orig_data[self.config.raft_validation_start : self.config.raft_validation_start + 10]
                assert len(orig_data) == 10
        else:
            if split == "validation":
                split = "test"
            orig_data = [example for example in self.orig_data[split]]
        for i, example in enumerate(orig_data):
            example["label"] = int(example["Label"]) - 1 if int(example["Label"]) > 0 else 0
            example["idx"] = example["ID"]
        return orig_data

    def compute_metric(self, accumulated):
        data = []
        idxs = accumulated["idx"]
        predictions = accumulated["prediction"]
        for idx, prediction in zip(idxs, predictions):
            data.append({"ID": idx, "Label": self.answer_choices[prediction]})
        result_df = pd.DataFrame(data=data, columns=["ID", "Label"]).astype({"ID": int, "Label": str})
        result_df.to_csv(self.config.dev_pred_file, index=False)
        matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}


def get_topics(model, count_vectorizer, n_top_words, tf_vectorizer):
    words = tf_vectorizer.get_feature_names()
    all_words = []
    for topic_idx, topic in enumerate(model.components_):
        all_words.append([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return all_words

def clean(document, stopwords, exclude, lemma):
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stopwords])
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized

def modify_templates(train_data, val_data, args):

    num_templates = args.num_templates

    preset_templates = list(dataset.dataset_stash)
    custom_templates = {}
    custom_answer_choices = []
    formatted_train = []


    for sample in train_data:
        sample_content = []
        for key, value in sample.items():
            if key == 'label' or key == "label_text" or not isinstance(value, str):
                continue
            else: 
                sample_content.append(value)
        sample_content = ' '.join(sample_content)
        formatted_train.append(''.join([x for x in sample_content if x.isalnum() or x == " "]))
    
    if args.retrieve_templates:
        import nltk
        nltk.download('wordnet') 
        from nltk.stem import WordNetLemmatizer 
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        nltk.download('omw-1.4') 
        import string
        from sentence_transformers import SentenceTransformer, util

        stopwords = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        lemma = WordNetLemmatizer()

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        top_k = 10

        num_classes = len(set([x['label'] for x in train_data]))    # val_data Just so we get the total number of classes
        word_count_per_class = {} 
        all_results = {x: None for x in range(num_classes)}
        all_words = []
        for cla in range(num_classes):
            instances = []
            for sample in train_data:
                if sample['label'] == cla:
                    sample_content = []
                    for key, value in sample.items():
                        if key == 'label' or key == "label_text" or not isinstance(value, str):
                            continue
                        else: 
                            sample_content.append(value)
                    sample_content = ' '.join(sample_content)
                    instances.append(sample_content)

            all_enc_sentences = []
            for cont in instances:
                query_enc = model.encode(cont, convert_to_tensor=True)
                all_enc_sentences.append(query_enc)
            

            current_doc = [clean(document, stopwords, exclude, lemma).split() for document in instances]
            current_doc_concat = list(set([item for  sublist in current_doc for item in sublist]))
            all_words += current_doc_concat

            results = []
            for cont in current_doc_concat:
                word_enc = model.encode(cont, convert_to_tensor=True)
                summed_score = 0
                for sent_enc in all_enc_sentences:
                    score = util.pytorch_cos_sim(word_enc, sent_enc)
                    summed_score += score.item()
                results.append((cont, summed_score))
            
            results = sorted(results, key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            all_results[cla] = results
        
        filtered_topic_words = [[] for k in range(num_classes)]
        topic_occurence_count = {x: 0 for x in all_words}
        for i, result in all_results.items():
            all_other_words = []
            for j, value in all_results.items():
                if i == j:
                    continue
                all_other_words += [x[0] for x in value]

            filtered_topic_words[i] = [x[0] for x in result if x[0] not in all_other_words]
            for word in result[:5]:
                topic_occurence_count[word[0]] +=1
        
        most_frequent_topics = sorted(topic_occurence_count, key=topic_occurence_count.get, reverse=True)

        outside_count = 0
        for i in range(num_classes):
            if len(filtered_topic_words[i]) == 0:
                if topic_occurence_count[most_frequent_topics[outside_count]] <=3:
                    custom_answer_choices.append("CLS" + str(i)) # Textual representation of class being the number if no freq. word occurs across classes
                else:
                    custom_answer_choices.append(most_frequent_topics[outside_count])
                    outside_count+=1
            else:
                custom_answer_choices.append(filtered_topic_words[i][0])
        
        print("Topic-specific answer choices are: {}".format(custom_answer_choices))    

        if 'anli-r1' in args.dataset:
            preset_templates = ["hi", "anli-r1"]
        elif 'anli-r2' in args.dataset:
            preset_templates = ["hi", "anli-r2"]
        elif 'anli-r3' in args.dataset:
            preset_templates = ["hi", "anli-r3"]
        elif 'wic' in args.dataset:
            preset_templates = ["hi", "wic"]
        elif "wsc" in args.dataset:
            preset_templates = ["hi", "wsc"]
        elif args.dataset in ["neurips_impact_statement_risks", "one_stop_english", "semiconductor_org_types", "tai_safety_research", "systematic_review_inclusion", "overruling", "tweet_eval_hate", "twitter_complaints", "ade_corpus_v2", "banking_77", "terms_of_service", "sentevalcr", "emotion", "enron_spam", "sst5", "amazon_counterfactual_en"]:
            preset_templates = ["hi", args.dataset]
        elif "cb" in args.dataset:
            preset_templates = ["hi", "cb"]
        else:
            preset_templates = ["hi", "rte"]
        custom_templates = []


    elif args.dataset  in "emotion":
        preset_templates = ["emotion", ""]
    elif args.dataset in "sentevalcr":
        preset_templates = ["SetFit", "SentEval-CR"]   
    elif args.dataset in "enron_spam":
        preset_templates = ["SetFit", "enron_spam"]
    elif args.dataset in "amazon_counterfactual_en":
        preset_templates = ["SetFit", "amazon_counterfactual_en"]    
    elif args.dataset in "sst5":
        preset_templates = ["SetFit", "sst5"]
    elif args.dataset in ["anli-r1", "anli-r2", "anli-r3"]:
        preset_templates.append('')

    return [train_data, preset_templates, custom_templates, num_templates, custom_answer_choices, formatted_train]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='dataset from RAFT benchmark', default='rte')
    parser.add_argument('--efficient_finetune', type=str, default='lora')
    parser.add_argument('--num_shot', type=int, default=16)
    parser.add_argument('--num_templates', type=int, default=30)
    parser.add_argument('--ignore_save_path', help='override save', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--few_shot_seed', default=100, type=int)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--backbone', type=str, default="bigscience/T0_3B")
    parser.add_argument('--no_validation', help='override save', action='store_true')
    parser.add_argument('--test_mode', help='override save', action='store_true')
    parser.add_argument('--test_split', help='override save', action='store_true')
    parser.add_argument('--large_mode', help='override save', action='store_true')
    parser.add_argument('--use_pretrained_weights', help='override save', action='store_true')
    parser.add_argument('--pretrained_checkpoint_name', type=str, default="t03b_ia3_finish.pt")
    parser.add_argument('--inference_split', type=int, default=0)
    parser.add_argument('--balanced_sampling', help='override save', action='store_true')


    parser.add_argument('--template', type=int, default=10)
    parser.add_argument('--majority_voting', help='override save', action='store_true')

    parser.add_argument("--randomize_ensemble_seed",  help='override save', action='store_true')

    parser.add_argument('--top_k_checkpoints', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--average_method', type=str, default="greedy_soup")

    parser.add_argument('--majority_baseline', help='override save', action='store_true')

    parser.add_argument('--retrieve_templates', help='override save', action='store_true')
    parser.add_argument("--tune_templates",  help='override save', action='store_true')
    parser.add_argument("--calibrate_templates",  help='override save', action='store_true')
    parser.add_argument('--num_retrieved_templates', type=int, default=5)
    parser.add_argument("--template_choice_mode", type=str, default="auto", choices=['dataset', 'auto', "auto_dataset", "random", "inverted"])
    parser.add_argument("--template_mode", type=str, default="auto", choices=['none', 'random', "worst_k", 'auto'])
    parser.add_argument("--template_kb", type=str, default="pretraining", choices=['all', "pretraining"])
    parser.add_argument("--restrict_choices", type=str, default="None", choices=['template-only', "topic-only"])

    parser.add_argument('--use_deepspeed', help='override save', action='store_true')
    parser.add_argument('--gradient_checkpointing', help='override save', action='store_true')

    parser.add_argument('--paper_comparison', help='override save', action='store_true')


    args = parser.parse_args()

    print("Dataset {}".format(args.dataset))
    print("Seed: {}".format(args.seed))
    print("Sample Seed: {}".format(args.few_shot_seed))

    config = Config(kwargs={'dataset': args.dataset, 'few_shot': True, 'num_shot' : args.num_shot, "few_shot_random_seed": args.few_shot_seed})
    config.balanced_sampling = args.balanced_sampling
    
    dataset = get_dataset_reader(config)

    train_data = dataset.read_few_shot_dataset()
    save_path = f'{args.experiment_name}_{args.dataset}_{args.seed}_{args.few_shot_seed}_{args.num_shot}_{args.efficient_finetune}'
        
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.test_split:
        val_data = dataset.read_orig_dataset('validation')
    else:
        val_data = dataset.read_orig_dataset('test')

    if args.dataset in ["amazon_counterfactual_en", "sentevalcr", "enron_spam"] and not args.test_split:
        val_data = [x for x in val_data if x not in train_data]
    
    if args.test_split and args.dataset in ["rte", "wic", "wsc", "cb"]:
        val_data_fixed = []
        for i, entry in enumerate(val_data):
            entry['label'] = 0
            val_data_fixed.append(entry)
        val_data = val_data_fixed

    save_path = os.path.join("experiments", args.experiment_name, save_path)
    

    if args.ignore_save_path and os.path.exists(save_path) and os.path.isdir(save_path):
        shutil.rmtree(save_path)

    train_data, preset_templates, custom_templates, num_templates, custom_answer_choices, formatted_train = modify_templates(train_data, val_data, args)
            
   
    if args.paper_comparison:
        if args.dataset == "rte" and args.large_mode: # Some inference complications, need to split RTE
            val_data = np.array(val_data)
            val_data_split = np.split(val_data, 2)[args.inference_split]
            val_data = val_data_split.tolist()

        train_data = pd.DataFrame(train_data)
        val_data = pd.DataFrame(val_data)

        if "idx" not in train_data:
            train_data['idx'] = range(1, len(train_data) + 1)
            val_data["idx"] = range(1, len(val_data) + 1)

        print("Training size", len(train_data))
        print("Validation siz", len(val_data))
    else:
        val_data_1 = val_data[:args.num_shot]
        val_data_2 = val_data[args.num_shot:]

        train_data = pd.DataFrame(train_data)
        val_data_1 = pd.DataFrame(val_data_1)
        val_data_2 = pd.DataFrame(val_data_2)

        if "idx" not in train_data:
            train_data['idx'] = range(1, len(train_data) + 1)
            val_data_1["idx"] = range(1, len(val_data_1) + 1)
            val_data_2["idx"] = range(1, len(val_data_2) + 1)


        print("Training size", len(train_data))
        print("Validation size 1", len(val_data_1))
        print("Validation size 2", len(val_data_2))


    average_method = "best"
    if args.top_k_checkpoints > 1:
        average_method = args.average_method

    if args.large_mode:
        backbone = "bigscience/T0pp"
    elif args.test_mode:
        backbone = "t5-small"
    else:
        backbone=args.backbone

    if args.paper_comparison:
        tuning_data = val_data
    else:
        tuning_data = val_data_1
    
    if backbone == "yuchenlin/BART0pp" or backbone == "yuchenlin/BART0":
        print("Loading BART0 preset...")
        presets = "few_shot_text_classification_bart0"
    elif "lora" in args.efficient_finetune:
        print("Loading LORA_IA3 preset...")
        presets = "few_shot_text_classification_lora"
    else:
        print("Loading IA3 preset...")
        presets = "few_shot_text_classification_ia3"

    import time
    start = time.time()

    predictor = MultiModalPredictor(label=label, path=save_path, eval_metric="accuracy", problem_type='classification',  verbosity=3).fit(
        train_data,
        tuning_data = tuning_data,
        seed = args.seed,
        dataset_name = args.dataset,
        restrict_choices  = args.restrict_choices,
        hyperparameters = {
         "data.templates.preset_templates" : preset_templates,
         "data.templates.tune_templates": args.tune_templates,
         "data.templates.custom_templates": custom_templates,
         "data.templates.num_templates" : num_templates,
         "data.templates.train_data": formatted_train,
         "data.templates.custom_answer_choices": custom_answer_choices,
         "data.templates.num_retrieved_templates": args.num_retrieved_templates,
         "data.templates.template_choice_mode": args.template_choice_mode,
         "data.templates.template_kb": args.template_kb,   
         "data.templates.template_mode": args.template_mode,
         "model.t_few.checkpoint_name": backbone,
         "model.t_few.gradient_checkpointing" : args.gradient_checkpointing,
         "model.t_few.calibrate_templates": args.calibrate_templates,
         "optimization.lora.use_pretrained_weights": args.use_pretrained_weights,
         "optimization.efficient_finetune":args.efficient_finetune,
         "optimization.lora.pretrained_checkpoint_name": args.pretrained_checkpoint_name,
         "model.t_few.majority_voting": args.majority_voting,
         "env.strategy": "deepspeed_stage_3" if args.use_deepspeed else "ddp_spawn",
         "optimization.patience" : 100,
         "optimization.check_val_every_n_epoch": args.check_val_every_n_epoch if not args.no_validation else 1000,
         "optimization.val_check_interval": 1.0,
         "optimization.top_k_average_method": average_method,
         "optimization.top_k" : args.top_k_checkpoints,
        },
         presets=presets,
    )

    end = time.time()
    train_time = end - start
    print("Elapsed time train: ", end - start)
    start_inference = time.time()

    if args.paper_comparison:
        y_pred, y_pred_probs = predictor.predict(data=val_data, seed=args.seed)
        print(set(val_data[label]))
        if len(set(val_data[label])) > 1:
            scores_mcc = matthews_corrcoef(val_data[label].to_numpy(), y_pred)
            scores = {'acc': (y_pred == val_data[label]).to_numpy().mean(), "mcc": scores_mcc}
            print(scores)
            with open(os.path.join(predictor.path, f'result_dev_mcc_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
                f_out.write("{}".format(str(scores['mcc'])))
            with open(os.path.join(predictor.path, f'result_dev_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
                f_out.write("{}".format(str(scores['acc'])))
        else:
            pred_path_dir = os.path.join("my-superglue-11b", "data", args.dataset)

            if args.dataset == "rte" and args.large_mode:
                pred_path = os.path.join(pred_path_dir, "predictions_" + str(args.inference_split) + ".csv")
            else:
                pred_path = os.path.join(pred_path_dir, "predictions.csv")

            if not os.path.exists(pred_path_dir):
                os.makedirs(pred_path_dir)

            with open(pred_path, "w") as f_out:
                for i, pred in enumerate(y_pred):
                    pred_dict = {"idx" : i, "label": dataset.orig_data.features["label"].names[pred]}
                    json.dump(pred_dict, f_out)
                    f_out.write("\n")
    else:
        y_pred, y_pred_probs = predictor.predict(data=val_data_2, seed=args.seed)
        if not all([x == -1 for x in val_data_2[label]]):
            scores = {'acc': (y_pred == val_data_2[label]).to_numpy().mean()}
            print(scores)
        with open(os.path.join(predictor.path, f'result_dev_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
            f_out.write("{}".format(str(scores['acc'])))

        np.savetxt(os.path.join(predictor.path, f'prediction_probs_dev_{args.dataset}_{str(args.seed)}.csv'), y_pred_probs, delimiter=",")
        y_pred.to_csv(os.path.join(predictor.path, f'prediction_dev_{args.dataset}_{str(args.seed)}.csv'))
    
    end_inference = time.time()
    inference_time = end_inference - start_inference

    with open(os.path.join(predictor.path, f'times_{args.dataset}_{str(args.seed)}.txt'), 'w') as f_out:
        f_out.write("{}\n".format(train_time))
        f_out.write("{}\n".format(inference_time))
    print("Elapsed time inference: ", end_inference - start_inference)
