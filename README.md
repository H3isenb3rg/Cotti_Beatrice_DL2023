# Deep Learning Project AA 2022/2023

This repository contains the code for our submission to the [I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started/overview) competition.

## Requirements

To run the notebook you need Python 3.7 or higher. It's recommended you use a virtual environment such as `venv` or `conda`.

The required packages are listed in `requirements.txt`. You can install them with:
```bash
pip install -r requirements.txt
```

To setup the precommit hooks, run:
```bash
pre-commit install
```

## Usage

Before running the script, you need to create a Kaggle API token. You can find the instructions [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

You can run the notebook using jupyter notebook:
```bash
jupyter notebook
```

Or you can run the compiled script with:
```bash
ipython DL_Beatrice_Cotti.py
```

To run on the Uni computer, you can use the `beatrice_cotti_dl.job` script (make sure you run the script from the project's root directory):
```bash
sbatch ./scripts/beatrice_cotti_dl.job
```
The job script uses the script version of the notebook (`DL_Beatrice_Cotti.py`), located in the project's root directory.
A pre-commit hook is provided to automatically convert the notebook to a script before each commit. To install it, run:
```bash
pre-commit install
```
You can do the conversion manually using the `convert_to_script.sh` script in the `scripts` dir:
```bash 
./scripts/convert_to_script.sh
```

## License

MIT, see `LICENSE` for more details.

## Authors

Matteo Beatrice, Luca Cotti