# Cascaded SVM for Three-Qubit Entanglement Classification

This research project investigates a cascaded support vector machine (SVM) framework for classifying three-qubit quantum states using a tunable number of Pauli observables.

## Overview

The framework exploits the nested convex structure of three-qubit entanglement classes:

- Fully separable states (S)
- Biseparable states excluding fully separable states (B \\ S)
- W-class states excluding biseparable states (W \\ B)
- GHZ-class states excluding W-class states (GHZ \\ W)

Three binary classifiers are arranged in a cascade to distinguish these classes. The project also investigates feature selection through a feature consensus approach, with the aim of reducing the number of input observables while retaining classification performance.

## Research scope

- Generation and labeling of pure and mixed three-qubit states
- Extraction of Pauli expectation values
- Training and evaluation of cascaded SVM classifiers
- Feature selection and reduced-observable classification
- Generalization and robustness evaluation

## Repository status

This repository is being initialized. Source code, dependency specifications, dataset generation instructions, and reproducibility details will be documented as they become available. No experimental results or installation commands are provided at this stage.

## Associated manuscript

*Cascaded Support Vector Machine for Three-Qubit Entanglement Classification Using Tunable Number of Measurements*

