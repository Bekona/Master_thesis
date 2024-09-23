# Master Thesis: Enhancing Machine Learning Performance in the Presence of Measurement Errors

This repository contains the code for my Master’s thesis titled "Enhancing Machine Learning Performance in the Presence of Measurement Errors". The thesis explores several approaches to handling label noise in machine learning, with a focus on three prominent methods: Unbiased Estimator, Label-Dependent Costs, and Importance Reweighting. These methods are implemented and evaluated on both synthetic datasets and real-world datasets, such as the UCI Breast Cancer dataset.

To run the experiments, you can open and execute the corresponding Jupyter notebooks. Each notebook corresponds to one of the three methods being tested:
- `unbiased_estimators.ipynb`: Implements and evaluates the Unbiased Estimator method.
- `label_dependent_costs.ipynb`: Implements and evaluates the Label-Dependent Costs method.
- `importance_reweighting.ipynb`: Implements and evaluates the Importance Reweighting method.

In the folder `uncorrelated_data` you can find the implementation of the methods using uncorrelated data.

References for these methods are:
- Natarajan, Nagarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari (2013), “Learning with noisy labels.
- Liu, Tongliang and Dacheng Tao (2016), “Classification with noisy labels by importance reweighting.”