# Requirements Document

## Introduction

This document specifies requirements for the Cell Cycle Web Frontend feature — a web-based interface and backend API that wraps the existing Cell Cycle Intelligence System (a Python ML pipeline for classifying cell cycle phases from BBBC048 fluorescence microscopy images). The feature also addresses known inconsistencies in the existing pipeline, adds proper documentation, and enables end-to-end testing. The system is a university project for IF3211 (Domain-Specific Computation) on the topic "Cell Cycle and Meiosis."

## Glossary

- **Pipeline**: The 3-layer ML system consisting of CNN classifier, ODE+HMM temporal validator, and Checkpoint anomaly detector
- **Backend_API**: A Python web server (Flask or FastAPI) exposing the Pipeline functionality as HTTP endpoints
- **Frontend**: A web-based user interface (HTML/CSS/JS) that communicates with the Backend_API
- **CNN_Classifier**: Layer 1 — a fine-tuned ResNet-18 model that classifies cell microscopy images into 7 cell cycle phases
- **ODE_Model**: Layer 2a — a Tyson-Novak Cyclin-CDK ordinary differential equation model that produces biologically-derived transition probabilities
- **HMM_Validator**: Layer 2b — a Hidden Markov Model that uses ODE-derived transition matrices to temporally smooth CNN predictions
- **Checkpoint_Detector**: Layer 3 — a rule-based anomaly detector that identifies biologically invalid phase transitions and durations
- **Population_Analyzer**: A module that aggregates single-cell predictions into population-level metrics (mitotic index, doubling time)
- **Grad_CAM**: Gradient-weighted Class Activation Mapping — an explainability technique showing which image regions the CNN attends to
- **BBBC048**: Broad Bioimage Benchmark Collection dataset 048 — Jurkat cell fluorescence microscopy images labeled by cell cycle phase
- **Phase**: One of 7 cell cycle stages: G1, S, G2, Prophase, Metaphase, Anaphase, Telophase
- **Data_Loader**: The module responsible for loading and splitting the BBBC048 dataset into train/val/test sets
- **Class_Weighting**: A technique to handle imbalanced datasets by assigning higher loss weights to underrepresented classes
- **Stratified_Split**: A dataset splitting method that preserves the class distribution ratio across train/val/test sets

## Requirements

### Requirement 1: Backend Pipeline Fixes — Data Loading and Training

**User Story:** As a developer, I want the data loading and model training code to be correct and honest, so that the system produces reliable results and the codebase is not misleading.

#### Acceptance Criteria

1. WHEN the Data_Loader module is renamed, THE Pipeline SHALL use the filename `bbbc048_loader.py` instead of `synthetic_data.py` to accurately reflect that the module loads real BBBC048 data
2. WHEN the Data_Loader splits the dataset, THE Data_Loader SHALL perform stratified splitting using the configured ratios (70% train, 15% validation, 15% test) such that each split's per-class proportion differs from the overall dataset proportion by no more than 5 percentage points
3. WHEN the CNN_Classifier computes training loss, THE CNN_Classifier SHALL apply inverse-frequency class weighting to the CrossEntropyLoss function, where each class weight equals the total sample count divided by the product of the number of classes and that class's sample count
4. THE CNN_Classifier SHALL train for a minimum of 20 epochs by default, as configured in the NUM_EPOCHS parameter
5. WHEN the model.py module is loaded, THE Pipeline SHALL import numpy at the top of the file (before any function or class definitions) rather than at the bottom
6. WHEN the Ground_truth.lst file is present in the data directory, THE Data_Loader SHALL parse it and validate that each loaded image's phase label matches the phase directory assignment in Ground_truth.lst
7. IF the dataset directory contains fewer than 1 image in any expected phase subdirectory, THEN THE Data_Loader SHALL raise an error indicating which phase subdirectory is empty or missing

### Requirement 2: Backend Pipeline Fixes — HMM Transparency

**User Story:** As a reviewer, I want the HMM layer to be transparent about its demonstration nature, so that the system does not misrepresent simulated results as real CNN-to-HMM integration.

#### Acceptance Criteria

1. THE HMM_Validator SHALL include a docstring at the module level and a comment above the sequence-generation function stating that the time-lapse sequence is simulated and not derived from actual sequential CNN predictions on real time-lapse data
2. WHEN the HMM comparison visualization is generated, THE Frontend SHALL display a disclaimer label stating "Demonstration: simulated time-lapse sequence (not real sequential predictions)" positioned directly above or below the visualization and visible without scrolling
3. THE Pipeline SHALL include the text "Conceptual Demonstration" in the title or heading of each HMM-related user-facing output, specifically: the HMM comparison plot title, the HMM section heading in the Frontend, and the HMM section heading in documentation files
4. WHEN the Backend_API returns HMM comparison results, THE Backend_API SHALL include a metadata field indicating that the HMM layer is a conceptual demonstration using simulated sequences

### Requirement 3: Backend Pipeline Fixes — Population Analysis

**User Story:** As a reviewer, I want the population analysis to be conceptually sound, so that treating independent test-set predictions as a population snapshot is properly contextualized.

#### Acceptance Criteria

1. WHEN the Population_Analyzer computes metrics from test-set predictions, THE Population_Analyzer SHALL include in its output a disclaimer field containing at minimum: (a) a statement that independent single-image classifications are treated as a proxy for a simultaneous population snapshot, and (b) at least one stated limitation: that no temporal ordering or cell-tracking links the images
2. WHEN population results are displayed, THE Frontend SHALL render the disclaimer text from the Population_Analyzer output on the population analysis panel, visible without requiring user interaction (e.g., not hidden behind a tooltip)
3. IF the Population_Analyzer receives fewer than 2 test-set predictions, THEN THE Population_Analyzer SHALL skip population metric computation and return a message indicating that insufficient data is available for population-level analysis

### Requirement 4: Backend API

**User Story:** As a frontend developer, I want a REST API that exposes all pipeline functionality, so that the web interface can trigger training, classification, and analysis operations.

#### Acceptance Criteria

1. WHEN a pipeline execution request is received, THE Backend_API SHALL start the full pipeline (train + evaluate + analyze) asynchronously and return a job status identifier within 2 seconds of the request
2. WHILE training is in progress, THE Backend_API SHALL expose an endpoint that returns the current epoch number, training loss, and training accuracy
3. IF a training progress request is received while no training is in progress, THEN THE Backend_API SHALL return a response indicating that no training session is active along with the completion status of the last run if available
4. WHEN a single microscopy image in PNG, JPG, or TIF format not exceeding 10 MB is uploaded via the API, THE Backend_API SHALL return the predicted phase, confidence scores for all 7 phases, and a Grad_CAM heatmap image
5. IF an uploaded image exceeds 10 MB or is not in PNG, JPG, or TIF format, THEN THE Backend_API SHALL reject the request with an error message indicating the accepted formats and size limit
6. THE Backend_API SHALL expose an endpoint to retrieve the latest pipeline results including: confusion matrix data, classification report, ODE dynamics data, HMM comparison data, anomaly list, and population summary
7. THE Backend_API SHALL expose an endpoint to retrieve generated plot images (training curves, confusion matrix, Grad-CAM, ODE dynamics, HMM comparison, anomaly timeline, population dashboard) in PNG format
8. IF the Backend_API receives a classification or results retrieval request while no trained model exists, THEN THE Backend_API SHALL return an error message indicating that the pipeline must be run first
9. THE Backend_API SHALL use CORS headers to allow requests from the Frontend served on a different port

### Requirement 5: Web Frontend — Dashboard and Navigation

**User Story:** As a user, I want a clean web dashboard that lets me run the pipeline and view all results in one place, so that I do not need to use the command line.

#### Acceptance Criteria

1. THE Frontend SHALL provide a main dashboard page with navigation to: Pipeline Control, Classification Results, ODE Dynamics, HMM Comparison, Anomaly Detection, Population Analysis, and Single Image Upload
2. WHEN the user clicks the "Run Pipeline" button and no pipeline execution is currently in progress, THE Frontend SHALL trigger full pipeline execution via the Backend_API and disable the button until execution completes or fails
3. WHILE the pipeline is running, THE Frontend SHALL poll the Backend_API at most every 2 seconds and display a progress indicator showing the current training epoch, total epochs, loss value, and accuracy value
4. WHEN the pipeline completes successfully, THE Frontend SHALL automatically load and display all result panels corresponding to the navigation sections listed in criterion 1
5. IF the pipeline execution fails or the Backend_API returns an error during execution, THEN THE Frontend SHALL display an error message indicating the failure reason and re-enable the "Run Pipeline" button
6. IF the user clicks the "Run Pipeline" button while a pipeline execution is already in progress, THEN THE Frontend SHALL keep the button disabled and display a message indicating that a pipeline run is already active

### Requirement 6: Web Frontend — Classification Results Panel

**User Story:** As a user, I want to see the CNN classification results clearly, so that I can evaluate model performance.

#### Acceptance Criteria

1. WHEN classification results are available, THE Frontend SHALL display the confusion matrix as a heatmap image with both axes labeled by the 7 phase names (G1, S, G2, Prophase, Metaphase, Anaphase, Telophase)
2. WHEN classification results are available, THE Frontend SHALL display per-class precision, recall, and F1-score in a table with columns for class name, precision, recall, F1-score, and support count, with metric values shown to 2 decimal places
3. WHEN classification results are available, THE Frontend SHALL display the overall test accuracy as a percentage value (to 2 decimal places) in a visually distinct summary section above or before the detailed metrics table
4. WHEN classification results are available, THE Frontend SHALL display the training curves as a plot showing both loss and accuracy on the y-axis against epoch number on the x-axis
5. IF classification results fail to load from the Backend_API, THEN THE Frontend SHALL display an error message indicating that results could not be retrieved and suggesting the user verify the pipeline has been run

### Requirement 7: Web Frontend — ODE and HMM Panels

**User Story:** As a user, I want to visualize the biological modeling layers, so that I can understand how biology constrains the predictions.

#### Acceptance Criteria

1. WHEN ODE results are available, THE Frontend SHALL display the Cyclin-CDK oscillator dynamics plot showing CycB, CDK1, and APC concentrations over time with labeled axes (time on x-axis, concentration on y-axis) and a legend identifying each curve
2. WHEN ODE results are available, THE Frontend SHALL display the ODE-derived transition probability matrix as a heatmap with the 7 phase labels on both axes
3. WHEN HMM results are available, THE Frontend SHALL display the ground truth, CNN-only, and HMM-corrected phase sequences as three aligned horizontal tracks so that the user can visually compare predictions at each time step
4. WHEN HMM results are available, THE Frontend SHALL display both the CNN-only accuracy and the HMM-corrected accuracy as percentages with at least one decimal place, along with the numerical difference between them
5. WHEN HMM results are displayed, THE Frontend SHALL show the demonstration disclaimer as specified in Requirement 2
6. IF ODE or HMM results are not yet available, THEN THE Frontend SHALL display a message indicating that the pipeline must be run first to generate biological modeling results

### Requirement 8: Web Frontend — Anomaly Detection and Population Panels

**User Story:** As a user, I want to see checkpoint anomalies and population metrics, so that I can interpret the biological significance of the results.

#### Acceptance Criteria

1. WHEN the Backend_API returns anomaly results containing one or more detected anomalies, THE Frontend SHALL display the checkpoint anomaly timeline visualization
2. WHEN the Backend_API returns anomaly results containing one or more detected anomalies, THE Frontend SHALL list each detected anomaly showing its severity level (e.g., low, medium, high), checkpoint type (e.g., G1/S, G2/M, spindle assembly), and biological interpretation text
3. IF the Backend_API returns anomaly results containing zero detected anomalies, THEN THE Frontend SHALL display a message indicating that no checkpoint anomalies were detected in the analyzed sequence
4. WHEN population results are returned from the Backend_API, THE Frontend SHALL display the phase distribution as a pie chart or bar chart showing percentage per phase, the mitotic index as a percentage, the growth fraction as a percentage, the estimated doubling time in hours, and the proliferation status label
5. WHEN population results are displayed, THE Frontend SHALL show the contextual limitation note as specified in Requirement 3

### Requirement 9: Web Frontend — Single Image Classification

**User Story:** As a user, I want to upload a single cell image and get an immediate classification with explanation, so that I can interactively test the model.

#### Acceptance Criteria

1. THE Frontend SHALL provide a file upload interface that accepts PNG, JPG, and TIF image formats with a maximum file size of 10 MB
2. WHEN a user uploads an image, THE Frontend SHALL display a loading indicator, send the image to the Backend_API, and upon response display the predicted phase with confidence percentage rounded to 1 decimal place
3. WHEN a classification result is returned, THE Frontend SHALL display a bar chart of confidence scores for all 7 phases
4. WHEN a classification result is returned, THE Frontend SHALL display the Grad_CAM heatmap overlay on the uploaded image
5. IF the user uploads an image before a model has been trained, THEN THE Frontend SHALL display a message instructing the user to run the pipeline first
6. IF the user selects a file that is not PNG, JPG, or TIF format, or exceeds 10 MB, THEN THE Frontend SHALL reject the upload and display an error message indicating the accepted formats and size limit
7. IF the Backend_API returns an error during classification, THEN THE Frontend SHALL hide the loading indicator and display an error message indicating that classification failed

### Requirement 10: End-to-End Testing

**User Story:** As a developer, I want automated tests that verify the full pipeline from backend to frontend, so that I can confirm the system works correctly after changes.

#### Acceptance Criteria

1. THE Pipeline SHALL include unit tests for: data loading with stratified split verification (asserting that each split preserves per-class ratios within 5% tolerance), class weight computation (asserting inverse-frequency weights match expected values for known class counts), CNN forward pass (asserting output tensor shape is [batch_size, 7] with values summing to 1.0 after softmax), ODE solver output shape and value ranges (asserting concentrations remain non-negative and do not exceed 2.0), HMM Viterbi decoding (asserting that a known observation sequence with known transition/emission matrices produces the expected state sequence), and checkpoint anomaly detection logic (asserting that a known invalid transition sequence produces the expected anomaly flags)
2. THE Pipeline unit tests SHALL execute without requiring the full BBBC048 dataset or a pre-trained model, using synthetic fixtures or minimal test data generated within the test setup
3. THE Pipeline SHALL include integration tests that verify all Backend_API endpoints defined in Requirement 4 return the expected HTTP status codes (200 for success, 400 for invalid input, 404 for missing resources) and response bodies containing the documented fields
4. THE Pipeline SHALL include an end-to-end test that runs the full pipeline on a subset of no more than 10 images per class and verifies that all output artifacts (model file, plot images, JSON results) are produced as non-empty files
5. WHEN tests are executed via a single command, THE Pipeline SHALL complete all unit and integration tests within 120 seconds (excluding end-to-end tests) and produce a test report indicating pass/fail status and execution duration for each test case
6. IF any test case fails, THEN THE Pipeline test report SHALL identify the failed test by name and include the assertion error message

### Requirement 11: Documentation

**User Story:** As a reviewer or new team member, I want comprehensive documentation explaining how the system works, so that I can understand the architecture, run the project, and evaluate the biological reasoning.

#### Acceptance Criteria

1. THE Pipeline SHALL include a README.md file at the repository root containing: project overview, architecture diagram (text-based), installation instructions with runnable example commands, usage instructions for both CLI and web interface, and dataset attribution
2. THE Pipeline SHALL include an ARCHITECTURE.md file explaining the 3-layer pipeline design, the biological reasoning behind each layer, and how the ODE model connects to the HMM
3. THE Pipeline SHALL include inline code comments in each module containing ODE, HMM, or checkpoint logic explaining the biological significance of key parameters (ODE constants, checkpoint thresholds, phase durations)
4. THE README.md SHALL explicitly state which components are fully functional (CNN classifier, ODE solver, checkpoint detector) and which are conceptual demonstrations (HMM on simulated time-lapse, population analysis assumptions)
5. THE Documentation SHALL include a section on known limitations and future work (real time-lapse integration, G0 detection, multi-cell tracking)
