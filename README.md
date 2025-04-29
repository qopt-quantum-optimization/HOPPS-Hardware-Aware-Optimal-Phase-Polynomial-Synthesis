# HOPPS — Hardware-Aware Optimal Phase Polynomial Synthesis with Blockwise Optimization

## Setup
- Create a Python environment:
  ```bash
  python3.9 -m venv Phase_Z3
  ```
- Activate the environment and install required packages:
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

Run the following scripts to perform different experiments:

- **For Table 2**:
  ```bash
  python run_MaxCut_Random.py
  ```

- **For Table 3**:
  ```bash
  python run_permuted_mapped.py
  ```

- **For Table 4 (first part)**:
  ```bash
  python run_MaxCut_Regular.py --name Qiskit
  ```
  > After completing the experiment with `--name Qiskit`, rerun the script with `--name ArPhase` and `--name 2QAN` to complete the full set of experiments.

- **For Table 4 (second part)**:
  ```bash
  python run_LABS.py --name Qiskit
  ```
  > After completing the experiment with `--name Qiskit`, rerun the script with `--name ArPhase`.

- **For Figure 7**:
  ```bash
  python run_blocks_size.py
  ```


## Example

Please refer to `demo.ipynb` for a general usage example.