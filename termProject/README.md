    

PA5
===

COSI 101A -- Brandeis University
================================

##### Professor Pengyu Hong

###### Date: 05/07/2022

* * *

### Description

This is our group's (Mason Ware, Nicole Meng, Novia Wu) final convolutional neural network used to classify handwriten mathematic expressions. This is our submission for the final term project. The model we implemented and decided to go with in the end, housed in `main.py`, utilizes pytorch and it's api. The details about the model, its layers, and its innerworkings can be found at `report.md`

The following are the main components of the program:

### Main Engine

* `main.py`

This is the main module of the program, any and all interaction with the model will be done through this file.

### Sub-Directories

* `data`

All data used for the training of the model is housed here.

* `utils`

All misc. scripts and parsers are housed here.

### Dependencies

1. colorama==0.4.4
2. keras==2.8.0
3. matplotlib==3.5.1
4. numpy==1.22.3
5. pandas==1.4.1
6. scikit_learn==1.0.2
7. tensorflow==2.8.0


All dependencies can be found in the `./requirements.txt` file. Moreover, they can be automatically installed using the shell command: `pip install -r requirements.txt` and the most up to date versions of the dependencies can be installed using the shell command: `pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U` .


* * *

Build & Run
-----------

### Build Instructions

[!If you would like to use the saved model located at `final_model.h5`, proceed to the next section] In order to build a final model, run the following commands:

1. `python3.9 main.py -t --dir <dir_name>`
If you would like to train the model on a dataset

2. `python3.9 main.py -s --dir <dir_name>`
If you would like to train and then save the model on a data set.

Also, run `python3.9 main.py -h` to see a list of flags that alter certain attributes of the outpuit & model.

### Run Instructions

In order to run a model, ensure that there is a `final_model.py` in the root directory of the project. Once that is done, run the command `python3.9 main.py -r --dir <dir_name>` to have the model predict the classes of each image within the provided directory. The model will output these predictions, per requirements, to a `results.csv` file.

* * *

_© 2022 MASON WARE_