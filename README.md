# Morpheus-Deblend

This repo generates the training data and the model for Morpheus-Deblend.

This is the active development repo for the project and as such you will find
many versions and experiments for different approaches to the problem.

I recommend you work in a virtual environment.

### Requirements
- `astropy`
- `comet_ml`
- `flow_vis`
- `gin`
- `numpy`
- `requests`
- `tensorflow==2.2`
- `tensorflow_addons`
- `scarlet` (https://github.com/pmelchior/scarlet)
- `scipy`
- `sharedmem`
- `scikit-image`
- `scikit-learn`
- `tqdm`

After the requirements are installed install the local repo:

`pip install -e .`

You can then download the data by running

`python src/data/make_dataset.py`

Then generate the training data using

`python src/features/build_features.py`

Then train a model by running

`python src/models/train_model.py src/models/morpheus-deblend.gin`

In order to run the training you need to have a comet.ml account setup
as that is where the training is logged.
