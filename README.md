# ProtoPGTN

## Datasets
All the datasets are available from [Time Series Classification website](https://www.timeseriesclassification.com/dataset.php). The provider formats the data, and files in `.ts` are used in this repository. All the datasets should be saved in `datasets` repository. There is a sample dataset called ArticularyWordRecognition in this folder, which can be used for testing. `dataset_configs.json` saved all the dataset information needed for running. 

## Environment Set Up
Please create an virtual environment in python, and install the `requirements.txt` provided. Python 3.12 is used in this project.

## Model Training and Testing
Simply training and testing model on sample dataset using `python3 protopgtn_model.py`. If other datasets will be tested, `--dataset_name` can be used to specify which dataset to run. All the available arguments are specified in `settings.py`. All the datasets are conducted with hyperparametr tuning and the hyperparameter selected for each dataset is stored in `additional_results/hyperparameter_result.csv`. The raw accuracy for both multivariate and univariate datasets is stored in `additional_results/accuracy_summary_multivariate.csv` and `additional_results/accuracy_summary_univarite.csv`, respectively. The raw training results and memory usage are also stored in this folder. 
## Interpretability Visualization
### Prototype Visualization
Please run `plot_prototypes.py` with dataset_name as arguments. The prototypes grouped by classes will be saved to `figs/prototypes`.
## Decision making process
Please run `plot_teting.py` with dataset_name as arguments. There is a variable `i` controling which test sample to draw. The most similar part to each prototype will be drawn and saved to `figs/testing/dataset_name/i`
## Comparable Models
The models for comparison are adpated from exisiting repositories. The links are: [TimesNet](https://github.com/thuml/TimesNet), [Zerveas](https://github.com/gzerveas/mvts_transformer), [MLP](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline), and [ProtoPLSTM](https://github.com/ilovesea/ProtoPLSTM). 
The adapted ProtoPLSTM and ProtoPGTN codes are stored in `ProtoPLSTM` folder, and these codes are adapted from [ProtoPLSTM](https://github.com/ilovesea/ProtoPLSTM).