#!/usr/bin/env

source ~/anaconda3/etc/profile.d/conda.sh

input_variable="taxonomy_classification"

ENVS=$(conda env list | awk '{print $input_variable}' )

if [[ $ENVS != *"$input_variable"* ]]; then

  echo "# BASH: conda env create # source activate phd
  name: $input_variable
  dependencies:
  - python=3.7.5
  - pytorch
  - pytorch-gpu
  - numpy
  - pandas
  - scipy
  - numpy
  - scikit-learn
  - pip
  - pip:
    - ktrain
    - bs4
    - dvc
    - pandarallel
    - flask
    - wget">environment.yml


  #list name of packages
  echo "installing base packages"
  conda env create -f=environment.yml
  conda activate $input_variable
  pip install -U spacy
  python -m spacy download en_core_web_sm



else
   echo "$input_variable already exist!"
   conda activate $input_variable
fi