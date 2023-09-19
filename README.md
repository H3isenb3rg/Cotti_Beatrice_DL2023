# Deep Learning Project AA 2022/2023

[I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started/overview)

## Requirements

To run the notebook you need Python 3.7 or higher.

The required packages are listed in `requirements.txt`. You can install them with:
```bash
pip install -r requirements.txt
```

It's recommended you use a virtual environment such as `venv` or `conda`.

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

To run on the Uni computer, you first need to convert the notebook to a script. You can do this with the `convert_to_script.sh` script in the `scripts` dir (make sure you run the script from the project's root directory):
```bash 
./scripts/convert_to_script.sh
```
This will create a `DL_Beatrice_Cotti.py` file in the project's root directory.
You can then run the script on the Uni computer with the `beatrice_cotti_dl.job` script:
```bash
sbatch ./scripts/beatrice_cotti_dl.job
```

A pre-commit hook is provided to automatically convert the notebook to a script before each commit. To install it, run:
```bash
pre-commit install
```

## License

MIT, see `LICENSE` for more details.

## Authors

Matteo Beatrice, Luca Cotti