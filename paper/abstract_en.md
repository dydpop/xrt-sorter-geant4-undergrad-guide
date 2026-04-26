# Abstract

This project designs and implements a Geant4-based simulation prototype for X-ray transmission mineral sorting. The system includes an X-ray source, ore samples, detector response, event-level data output, and Python-based analysis. Configuration files are used to describe materials, source settings, and spectrum inputs.

The analysis workflow extracts interpretable features such as transmission rate, energy deposition, direct-hit counts, and scatter-related signals. Threshold-based classification and Logistic Regression are used for a coarse absorption-group classification task. In the current simulated dataset and task setting, the best accuracy is `0.98`.

The result supports an undergraduate-level conclusion: the simulation system can produce physically meaningful features for baseline classification validation. It should not be interpreted as evidence for all materials, all equipment settings, or real-world deployment.
