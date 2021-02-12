# Repo for the manuscript: Autonomous AFM imaging of single DNA molecules enabled by deep learning
This repo contains the code, models, and images to train and validate these models, used in the manuscript "Enabling automated AFM imaging of single molecules with deep learning".

A more comprehensive guide for the repo will be provided upon publication of the manuscript.

## Getting started
* Tested with Python 3.6.0.
* Install required packages. A list of our environment is provided in requirements.txt. We could not access the Nanoscope COM server from a virtual environment or container. Thus, be aware that some of the listed packages might not be needed.
* If you want to image plasmid DNA molecules, uncompress the models (available in the Models folder) in the same folder as the Python scripts. You could also use your own models for other type of molecules. Jut name them in the same way as we do (or change the name in plasmidAnalysis.py). In the folder Images, we also provide the images used to train and test our YOLOv3 and Siamese Network models, so you could train new models as well.
* Set scanning parameters in scanParameters.json
* Open Nanoscope and command line as administrator.
* From the command line, just go to the folder where the python scripts are and run:
    python runAFM.py -i scanParameters.json

## Contributors
[Javier Sotres](https://www.jsotres.com/)

[Juan F. Gonzalez-Martinez](https://github.com/juanfran2018)
