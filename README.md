# Enhancing Signal Stability in Untargeted GC-IMS Metabolomics Removing Batch and Time Effects via Orthogonal Projections

This repository contains the code and data processing pipelines for the updated version of the paper: *"Enhancing Signal Stability in Untargeted GC-IMS Metabolomics: Removing Batch and Time Effects via Orthogonal Projections"* (2026).

## Repository Structure and Contents
* `data/`: Contains the empirical raw matrices and metadata files (`peak_table_var.csv`, etc.).
* `notebooks/`: Jupyter notebooks containing the evaluation workflows, hyperparameter tuning, and statistical analysis.
* `src/`: Python source code modules.
    * `models.py`: Implementations of `EPOCorrector`, `LOESSCorrector`, and `SERRFCorrector`.
    * `synthetic_engine.py`: Engine for generating semi-synthetic drift scenarios.
    * `utils.py`: Data loading and formatting utilities.
* `docs/figures/`: Auto-generated output plots and statistical visualizations.

## Environment Setup
This project was developed using Python. To reproduce the analysis, create a virtual environment and install the required dependencies:

```bash
# 1. Clone the repository
git clone [https://github.com/adrianrefe4/gcims-workflows-2026.git](https://github.com/adrianrefe4/gcims-workflows-2026.git)
cd gcims-workflows-2026

# 2. Create a virtual environment named 'venv'
python -m venv venv

# 3. Activate the virtual environment:
# - On Linux / macOS:
source venv/bin/activate
# - On Windows (Command Prompt):
venv\Scripts\activate.bat
# - On Windows (PowerShell):
venv\Scripts\Activate.ps1

# 4. Install required dependencies
pip install -r requirements.txt
```

## Computational Environment & Hardware
The experiments and grid searches were executed under the following hardware configuration:
* **Operating System:** Windows (64-bit)
* **Processor:** Intel Core i7
* **Python Version:** Python 3.10+
*(Note: All data processing and pipeline evaluations were executed exclusively on the CPU: no GPU needed).*

---

## AI Usage Disclosure
During the preparation of this work, AI-based coding assistants (such as Anthropic Claude, Google Gemini and OpenAI Codex) were utilized to assist with Python code refactoring, structural optimization, and documentation. All algorithmic implementations, experimental designs and visualizations were reviewed, tested and validated by the author.

---

## License
This project is licensed under the terms of the MIT License. See the `LICENSE.md` file for details.

--- 

## Author
**Adrián Redondo Fernández**  
*Master in Health Data Science (Universitat Rovira i Virgili et al.) & Signal and Information Processing for Sensing Systems (IBEC)*  
Barcelona, 2026
