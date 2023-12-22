# bowling-simulation

(Timur Kesdogan & Tobia Ochsner)

## 👋 Quick Start

After cloning into this repository, set up a venv environment and install the python requirements using the following commands (on Windows, the commands to initialize the venv environment might look slightly different):

```
python3.10 -m venv env
source env/bin/activate
make init
```

If you installed new requirements, run the following command to update the `requirements.txt` file:

```
pip freeze > requirements.txt
```

## 🎳 Run the Simulation

Use the command `make run` to run the simulation. The simulation is automatically cached in the `cache` folder. Use the command `make run-cached` to rerun the last simulation or `make run-cached f=<file_name>` to run a specific cached simulation.

## Tests

We have sanity checks of the projective dynamics solver to test the effect of extenral forces or to test invarinces like the translation and rotation invariance of the shape and volume contraints. To run the tests, use the command `make test`.

## Features

- Projective Dynamics Solver
- 2D Simplicial Constraints
- Volume Constraints
- Collision Detection & Collision Constraints
- Faster batched 2D Simplicial Constraints
- (Incomplete) GPU version (`make run-gpu`)
  - Uses PyTorch for everything
  - Way faster collision detection

## 📚 Sources

- [Original Paper](https://users.cs.utah.edu/~ladislav/bouaziz14projective/bouaziz14projective.pdf)
- [Thesis with better explanation](https://purehost.bath.ac.uk/ws/portalfiles/portal/187951440/clewin_thesis.pdf)
- [Implementation](https://github.com/taichi-dev/meshtaichi/tree/main/projective_dynamics)
- [Another Implementation](https://github.com/fangqiaohu/ProjectiveDynamics/blob/master/muti_thre.py)
