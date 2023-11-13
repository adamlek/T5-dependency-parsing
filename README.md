# T5-dependency-parsing
Experimental model and fine-tuning on generative dependency parsing with T5

`conllu.py`: parse conllu files, a copy of the PyPi package (but it's not included in nixpkgs)

`data.py`: read the data, find mean hierarchical and lineal distance

`curriculum.py`: dataset for curriculum learning

`main.py`: train and validate the model

`torch-shell.nix`: create enviroment for the server

# Also included:
`Transformer/`: the transformer I implemented myself
