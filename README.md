# Enhancing Signal Stability in Untargeted GC-IMS Metabolomics Removing Batch and Time Effects via Orthogonal Projections

This repository contains the code and data processing pipelines for the updated version of the paper: *"Enhancing Signal Stability in Untargeted GC-IMS Metabolomics: Removing Batch and Time Effects via Orthogonal Projections"* (2026).

## Repository Structure and Contents
* `data/`: Empirical raw matrices and metadata files (`peak_table_var.csv`, `peak_table_pool.csv`, `annotations_pool.csv`).
* `docs/`: Evaluation workflows and analysis outputs:
  * `figures/`: Saved visualizations (high-resolution PNG/PDF plots).
  * `grid_results/`: Exported benchmarks and intermediate simulation outputs.
  * `Enhancing_Signal_Stability_in_untargeted_GCIMS_Metabolomics_Removing_Batch_and_Time_Effects_via_Orthogonal_Projections.ipynb`: Main analysis and demonstration notebook.
  * `df_master.csv`: Aggregated master metrics and benchmark results table.
* `src/`: Python source code modules.
    * `models.py`: Implementations of the transductive form for `ISFO` and `INSOC`, `LOESSCorrector`, and `SERRFCorrector`.
    * `synthetic_engine.py`: Engine for generating semi-synthetic data from our real pooled urine samples.
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

# 3. Install required dependencies
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
**AdriÃ¡n Redondo FernÃ¡ndez**  
*Master in Health Data Science (Universitat Rovira i Virgili et al.) & Signal and Information Processing for Sensing Systems (IBEC)*  
Barcelona, 2026
