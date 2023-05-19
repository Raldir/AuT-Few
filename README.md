## AuT-Few: Automated Few-shot Classification with Instruction-Finetuned Language Models
 
 This repository contains the code to replicate the main results presented in the paper: [Automated Few-shot Classification with Instruction-Finetuned Language Models](XXX).
> A particularly successful class of approaches for few-shot learning combines language models with prompts --- hand-crafted task descriptions that complement data samples. 
However, designing prompts by hand for each task commonly requires domain knowledge and substantial guesswork. We observe, in the context of classification tasks, that instruction finetuned language models exhibit remarkable prompt robustness, and we subsequently propose a simple method to eliminate the need for handcrafted prompts, named AuT-Few. This approach consists of (i) a prompt retrieval module that selects suitable task instructions from the instruction-tuning knowledge base, and (ii) the generation of two distinct, semantically meaningful, class descriptions and a selection mechanism via cross-validation. Over 12 datasets, spanning $8$ classification tasks, we show that AuT-Few outperforms current state-of-the-art few-shot learning methods. Moreover, AuT-Few is the best ranking method across datasets on the RAFT few-shot benchmark. Notably, these results are achieved without task-specific handcrafted prompts on unseen tasks.

### Installation
The code is extending a specific version of Autogluon, an AutoML library (LINK HERE), to implement AuT-Few. The following scripts install packages required for Autogluon, as well as packages specific for AuT-Few.
```
# Run following scripts from the terminal:
# ./t_few/init_environment.sh
# ./t_few/activate_environment.sh 
# ./t_few/setup_missing_libs.sh
# ./full_install.sh
# ./t_few/install_spacy.sh
```
Additionally, it is necessary to specify the environment variable `HF_HOME`, i.e. where the hugginface_hub will locally store data.

## Example

To run AuT-Few on the classification datasets, simply run the following script. Datasets and model pre-trained model weights are loaded automatically: 

```
cd experiments
./autfew.sh 32 0 
```
The first argument specifies the number of samples to use for training, the second argument specifies which GPU to use (multi-GPU training has not been tested). The script will train and evaluate AuT-Few on each dataset for five different seeds. If you wish to evaluate AuT-Few on one specific dataset, simply modify the shell script.

Similarly, to run the T-Few baseline, run the script:

```
./tfew.sh 0
```

## Citation
If you use AuT-Few in a scientific publication, please cite the following paper:

XXX

BibTeX entry:

```bibtex
@inproceedings{agmultimodaltext,
  title={Multimodal AutoML on Structured Tables with Text Fields},
  author={Shi, Xingjian and Mueller, Jonas and Erickson, Nick and Li, Mu and Smola, Alex},
  booktitle={8th ICML Workshop on Automated Machine Learning (AutoML)},
  year={2021}
}
```


## License

This library is licensed under the Apache 2.0 License.
