# Photodiode Event Detection

A Python signal-processing pipeline for detecting photodiode state transitions
in electrophysiological recordings.

I developed this pipeline as part of my neuroengineering research at UC Davis.
The goal was to reliably recover experimental event timing from noisy
photodiode signals and validate the detected events against reference states.

Research datasets and generated outputs are intentionally excluded from this
public repository.

## Problem

Neurophysiology experiments require accurate synchronization between recorded
neural activity and events presented during an experiment.

A photodiode provides a signal indicating when visual stimuli change, but the
recorded signal can contain noise, interference, and recording-specific
variation. A simple fixed threshold therefore does not always reliably recover
the underlying ON/OFF states.

The goal of this project was to build a robust algorithm that converts these
raw photodiode recordings into clean event states and transition timestamps.

## Pipeline

The detection pipeline performs the following steps:

1. Load the recorded photodiode signal and sampling frequency.
2. Filter unwanted signal interference when necessary.
3. Smooth the signal to reduce high-frequency noise.
4. Normalize the signal so thresholds can be applied consistently.
5. Detect ON and OFF states using configurable thresholds.
6. Identify state transitions and convert their indices to timestamps.
7. Apply event/refractory logic to suppress spurious transitions.
8. Compare detected states with reference states when ground truth is available.
9. Batch-test the algorithm across recordings and report detection accuracy.

## Threshold and Parameter Search

Different recordings can have different signal characteristics. I therefore
built utilities for testing combinations of detection thresholds and filtering
parameters.

For each configuration, the pipeline:

- runs the detector on the recording,
- compares the predicted photodiode state with the reference state,
- calculates sample-level accuracy,
- and identifies the best-performing configuration.

This allowed detection parameters to be selected quantitatively rather than
through manual inspection alone.

## Repository Structure

### `detect_photodiode_neuralynx.py`
Core event-detection implementation for Neuralynx recordings. Contains the
signal-processing and ON/OFF transition detection logic.

### `photodiode.py`
Photodiode processing utilities used by the pipeline.

### `batch_test_neuralynx.py`
Runs the Neuralynx detector across multiple recordings and calculates
accuracy and mismatch statistics.

### `batch_test_edf.py`
Batch evaluation workflow for EDF-derived recordings.

### `sweep_neuralynx.py`
Searches different threshold configurations and evaluates their detection
accuracy.

### `test_neuralynx.py`
Testing and validation utilities for Neuralynx recordings.

### `test_photodiode.py`
Tests for the photodiode processing workflow.

### `save_all_photodiode_outputs.py`
Utility for processing recordings and saving generated photodiode event
outputs.

## Technologies

- Python
- NumPy
- SciPy
- Digital signal processing
- Threshold-based event detection
- Batch evaluation and parameter search
- MATLAB `.mat` interoperability

## Validation

Detection performance is evaluated by comparing the reconstructed photodiode
state against reference states on a sample-by-sample basis:

    accuracy = mean(predicted_state == reference_state)

The batch-testing tools also report mismatches and detected ON/OFF event counts,
making it possible to identify recordings or parameter settings that require
additional investigation.

## Research Context

This work was developed for neuroengineering research involving
electrophysiological recordings. The public repository contains only the
software implementation.

Experimental recordings, research datasets, generated `.mat` outputs, and
other non-public research materials are excluded.

## Author

Sreyash Ravinuthala  
Computer Science, UC Davis
