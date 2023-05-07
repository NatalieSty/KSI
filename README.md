# KSI framework
This repository contains codes for Knowledge Source Intergration (KSI) framework in the paper:
**Bai, T., Vucetic, S., Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources, The Web Conference (WWW'19), 2019.**
This projec is to reproduce the results presented in the paper, explore the methodologies and possibly provide some potential improvements. You will also find a picture of the table of results, which are our reproducibility results. 
## Related Links:
* [Link to original paper](https://dl.acm.org/doi/10.1145/3308558.3313485)
* [Link to original github repo](https://github.com/tiantiantu/KSI)

## Dependencies ##
I used the following environment for the implementation:
* python==3.7.0
* torch==0.4.1
* numpy==1.15.1
* sklearn==0.19.2

## Input files ##
Before running the program, you need to apply for [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) dataset and put two files "NOTEEVENTS.csv" and "DIAGNOSES_ICD.csv" under the same folder of the project.

Once you get these two files, run preprocessing scripts `python preprocessing1.py`, `python preprocessing2.py`, `python preprocessing3.py` in order.

After running three preprocessing files, you can run any of four models ("python KSI_LSTM.py", "python KSI_LSTMatt.py", "python KSI_CNN.py", "python KSI_CAML.py") to see how much improvement KSI framework brings to the specific model.
