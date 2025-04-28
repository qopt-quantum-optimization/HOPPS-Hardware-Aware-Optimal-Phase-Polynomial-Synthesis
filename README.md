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
Run the following scripts for different experiments:
```bash
python run_MaxCut_Random.py
python run_permuted_mapped.py
python run_MaxCut_Regular.py --name Qiskit
python run_LABS.py --name Qiskit
```
