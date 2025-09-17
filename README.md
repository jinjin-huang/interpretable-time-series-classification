# Interpretable Time Series Dataset
This repository is for "ProtoPGTN: A Scalable Prototype-based Gated Transformer
Network for Interpretable Time Series Classification". We introduce ProtoPGTN, adapting Gated Transformer Network to extract temporal and spatial features for prototype learning as interpretability methods.
## Datasets
All the datasets are available from [Time Series Classification website](https://www.timeseriesclassification.com/dataset.php). The provider formats the data, and files in `.ts` are used in this repository. All the datasets should be saved in `datasets` repository. There is a sample dataset called ArticularyWordRecognition in this folder, which can be used for testing. `dataset_configs.json` in `src/utils` saved all the dataset information needed for running. 

## Environment Set Up
Please create an virtual environment in python, and install the `requirements.txt` provided. Python 3.12 is used in this project.

## ProtoPGTN Training and Testing
Simply training and testing model on sample dataset using `python3 -m src.ProtoPGTN.protopgtn_model.py` in the root. If other datasets will be tested, `--dataset_name` can be used to specify which dataset to run. All the available arguments are specified in `settings.py`, and can also be viewed through `python3 -m src.ProtoPGTN.protopgtn_model.py -h`. All the datasets are conducted with hyperparametr tuning and the hyperparameter selected for each dataset is stored in `archived_results/hyperparameter_result.csv`. These hyperparameter can be used by setting the arguments.The raw accuracy for both multivariate and univariate datasets is stored in `archived_results/accuracy_summary_multivariate.csv` and `archived_results/accuracy_summary_univarite.csv`, respectively. The raw training results and memory usage are also stored in this folder. 
## Interpretability Visualization
### Prototype Visualization
Please run `python3 -m src.ProtoPGTN.plot_prototypes.py` with dataset_name as arguments. The prototypes grouped by classes will be saved to `results/protopgtn/figures/prototypes`.
## Decision making process
Please run `plot_teting.py` with dataset_name as arguments. There is a variable `i` controling which test sample to draw. The most similar part to each prototype will be drawn and saved to `results/protopgtn/figures/testing/dataset_name/i`

## Model Comparison 
- **ProtoPLSTM**：The codes are adapted from https://github.com/ilovesea/ProtoPLSTM, and the adapted codes are stored in `src/ProtoPLSTM`. Similarity, run `python3 -m src.ProtoPLSTM.protoplstm_model.py` will start the training process, and results are saved in `results/protoplstm`.
- **ProtoPGTN**: Codes are adapted from ProtoPLSTM with only convolutional layers before prototype layer. Codes are saved in `src/ProtoPConv`, run exactly the same way as previous models.
- **TimesNet**: Clone the repository from https://github.com/thuml/Time-Series-Library. Set up the python environment by downloading the `requirements.txt`. Run `bash ./scripts/classification/TimesNet.sh` for classification.
- **Zerveas**: Clone the repository from https://github.com/gzerveas/mvts_transformer. Set up the environment using `requirements.txt`, and run classification using `python src/main.py --output_dir experiments --comment "classification from Scratch" --name $1_fromScratch --records_file Classification_records.xls --data_dir path/to/Datasets/Classification/$1/ --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 400 --lr 0.001 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy`. 
- **MLP**: Clone the repository from https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline. Run `MLP.py` for MLP classification results.

## Ablation Anlysis
Two ablation tests are conducetd. One is using GTN directly for classification, and the other is replacing cosine similarity with Euclidean distance. Codes are all stored in `src/Ablation_test`. Run same way as ProtoPGTN and the results are stored in `results/Ablation_test`. 

## Archived Results
All the raw results are saved in `archived_results`. 

