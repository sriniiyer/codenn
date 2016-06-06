

**Run CODENN**

Requirements

> Torch (http://torch.ch/docs/getting-started.html)
> Cutorch
> antlr4 for parsing C# (pip install antlr4-python2-runtime)

Setup environment

> export PYTHONPATH=~/codenn/src/:~/codenn/src/sqlparse

> export CODENN_DIR=~/codenn/

> export CODENN_WORK=./workdir

Build both csharp and sql datasets

Install modified sqlparse

> cd src/sqlparse/

> sudo python setup.py install

Build datasets

> cd src/model

> ./buildData.sh

Train codenn models and predict on test set

> ./run.sh {sql|csharp}
