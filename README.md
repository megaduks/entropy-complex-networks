## Entropy of Complex Networks

#### Scientific project funded by the Polish National Science Center 

Project description, publications, data, and code are described at [project's website](https://megaduks.github.io/entropy-complex-networks/)

### Using Jupytext for working with notebooks

We mostly use Jupyter Notebooks environment for working on experiments.
However, the experiments are stored in the repository using `percent` format. 
It has the following benefits:
* The changes are easy to track and review in pull requests,
* The files are valid python scripts (unlike notebooks) so all the IDE support
like code analysis, refactoring etc. works out of the box,
* `percent` format is compatible with a wide range of environments which is
especially helpful when team members want to use different environments.
The format is supported by:
    * Spyder IDE,
    * Hydrogen, a notebook editor based on Atom,
    * VS Code when using the vscodeJupyter extension,
    * Python Tools for Visual Studio,
    * and PyCharm Professional
    
Follow the [instructions](https://jupytext.readthedocs.io/en/latest/install.html) to install Jupytext.

Jupytext tool automatically convert paired notebooks to and from `percent` format
once the notebooks are paired. To pair all the scripts and to generate notebooks
from them the first time, run the script `pair_notebooks.py`.
The script needs to run only once after cloning clean repository.

