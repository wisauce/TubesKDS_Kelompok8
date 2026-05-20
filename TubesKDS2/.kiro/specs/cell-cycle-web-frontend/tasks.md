# Implementation Plan: Cell Cycle Web Frontend

## Overview

This plan implements the Cell Cycle Web Frontend feature in incremental steps: first fixing the backend pipeline (data loading, HMM transparency, population analysis), then building the FastAPI backend API, then the vanilla HTML/CSS/JS web frontend, followed by testing and documentation. Each task builds on previous ones, ensuring no orphaned code.

## Tasks

- [ ] 1. Backend pipeline fixes — Data loading and training
  - [x] 1.1 Rename `synthetic_data.py` to `bbbc048_loader.py` and update all imports
    - Rename the file in the project
    - Update all import statements across the codebase that reference `synthetic_data`
    - Ensure the module docstring reflects that it loads real BBBC048 data
    - _Requirements: 1.1_

  - [ ] 1.2 Implement stratified dataset splitting in `bbbc048_loader.py`
    - Use `sklearn.model_selection.train_test_split` with `stratify` parameter
    - Split into 70% train, 15% validation, 15% test
    - Ensure per-class proportion in each split differs from overall by no more than 5 percentage points
    - _Requirements: 1.2_

  - [ ] 1.3 Add inverse-frequency class weighting to CNN training in `model.py`
    - Compute class weights as `total_samples / (num_classes * count_i)` for each class
    - Pass weights to `torch.nn.CrossEntropyLoss(weight=...)`
    - Move numpy import to top of file
    - Set `NUM_EPOCHS = 20` as default
    - _Requirements: 1.3, 1.4, 1.5_

  - [ ] 1.4 Implement Ground_truth.lst validation in `bbbc048_loader.py`
    - Parse `Ground_truth.lst` file mapping filenames to phase labels
    - Validate each loaded image's phase label matches its directory assignment
    - Raise `ValueError` listing mismatches if any are found
    - Raise error if any expected phase subdirectory has fewer than 1 image
    - _Requirements: 1.6, 1.7_

- [ ] 2. Backend pipeline fixes — HMM transparency
  - [ ] 2.1 Add transparency docstrings and disclaimers to `hmm.py`
    - Add module-level docstring stating the time-lapse sequence is simulated
    - Add comment above sequence-generation function explaining demonstration nature
    - Include "Conceptual Demonstration" in HMM comparison plot title
    - Add `is_demonstration: bool = True` field to HMM output data
    - _Requirements: 2.1, 2.3, 2.4_

- [ ] 3. Backend pipeline fixes — Population analysis
  - [ ] 3.1 Add disclaimer and minimum-count guard to `population_analysis.py`
    - Add disclaimer field to output containing: (a) statement about independent classifications as proxy for population snapshot, (b) limitation that no temporal ordering or cell-tracking links the images
    - Add guard: if fewer than 2 predictions, skip computation and return insufficient-data message
    - _Requirements: 3.1, 3.3_

- [ ] 4. Checkpoint — Verify pipeline fixes
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Backend API — Project structure and schemas
  - [x] 5.1 Create API directory structure and install dependencies
    - Create `api/` directory with `__init__.py`, `server.py`, `routes/`, `services/`, `models/` subdirectories
    - Create `requirements.txt` or update existing with: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `httpx`
    - _Requirements: 4.1_

  - [ ] 5.2 Define Pydantic request/response schemas in `api/models/schemas.py`
    - Implement all schemas: `JobStatus`, `PipelineRunResponse`, `ProgressResponse`, `ClassificationResult`, `AnomalyItem`, `PopulationResult`, `HMMResult`, `PipelineResults`, `ErrorResponse`
    - _Requirements: 4.1, 4.2, 4.4, 4.6_

- [ ] 6. Backend API — Core endpoints
  - [ ] 6.1 Implement FastAPI server with CORS and static file serving in `api/server.py`
    - Create FastAPI app instance
    - Configure CORS middleware to allow frontend requests from different port
    - Mount static file serving for plots directory
    - Include route modules
    - _Requirements: 4.9_

  - [ ] 6.2 Implement pipeline runner service in `api/services/pipeline_runner.py`
    - Create `PipelineState` dataclass for in-memory state tracking
    - Implement background thread execution of the full pipeline
    - Update progress state (epoch, loss, accuracy) during training
    - Save results to `output/results/pipeline_results.json` on completion
    - _Requirements: 4.1, 4.2_

  - [ ] 6.3 Implement pipeline routes in `api/routes/pipeline.py`
    - `POST /api/pipeline/run` — start pipeline asynchronously, return job_id within 2 seconds, return 409 if already running
    - `GET /api/pipeline/progress` — return current epoch, loss, accuracy, or idle status with last run info
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 6.4 Implement results routes in `api/routes/results.py`
    - `GET /api/results` — return latest pipeline results JSON, or 404 if no results
    - `GET /api/plots/{name}` — serve plot PNG from output directory, or 404 if not found
    - `GET /api/health` — return `{"status": "ok"}`
    - _Requirements: 4.6, 4.7, 4.8_

  - [ ] 6.5 Implement classify route in `api/routes/classify.py`
    - `POST /api/classify` — accept image upload, validate format (PNG/JPG/TIF) and size (≤10 MB)
    - Run inference with trained model, generate Grad-CAM heatmap
    - Return predicted phase, confidence scores, gradcam_url
    - Return 400 for invalid file, 404 if no trained model exists
    - _Requirements: 4.4, 4.5, 4.8_

  - [ ] 6.6 Implement classifier service in `api/services/classifier.py`
    - Load trained model from `output/models/best_model.pth`
    - Implement image preprocessing (resize, normalize)
    - Implement Grad-CAM generation and saving
    - Implement file validation logic (format + size checks)
    - _Requirements: 4.4, 4.5_

- [ ] 7. Checkpoint — Verify backend API
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Web frontend — Layout and navigation
  - [ ] 8.1 Create `web/index.html` with main layout and navigation
    - Create `web/` directory structure: `css/`, `js/`, `assets/`
    - Build HTML page with navigation bar linking to: Pipeline Control, Classification Results, ODE Dynamics, HMM Comparison, Anomaly Detection, Population Analysis, Single Image Upload
    - Create panel container divs for each section
    - _Requirements: 5.1_

  - [ ] 8.2 Create `web/css/styles.css` with dark-themed responsive layout
    - Implement dark theme color scheme
    - Style navigation bar, panel containers, buttons, tables, alerts
    - Add responsive breakpoints for different screen sizes
    - Style progress indicators, loading states, error banners
    - _Requirements: 5.1_

- [ ] 9. Web frontend — Core application logic
  - [ ] 9.1 Implement `web/js/app.js` — navigation state and API communication
    - Implement panel switching (show/hide sections based on nav clicks)
    - Create API helper functions (fetch wrapper with error handling)
    - Implement network error display with retry button
    - Handle 3 consecutive poll failures → stop polling, show connection error
    - _Requirements: 5.1, 5.5_

  - [ ] 9.2 Implement `web/js/pipeline.js` — pipeline control panel
    - "Run Pipeline" button that triggers `POST /api/pipeline/run`
    - Disable button while pipeline is running, show "already active" message
    - Poll `GET /api/pipeline/progress` every 2 seconds
    - Display progress: current epoch, total epochs, loss, accuracy
    - On completion: auto-load all result panels
    - On failure: show error message, re-enable button
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 10. Web frontend — Result panels
  - [ ] 10.1 Implement `web/js/results.js` — classification results panel
    - Display confusion matrix heatmap image (from `/api/plots/confusion_matrix`)
    - Display per-class metrics table (precision, recall, F1, support) to 2 decimal places
    - Display overall test accuracy as percentage (2 decimal places) in summary section
    - Display training curves plot
    - Show error message if results fail to load
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 10.2 Implement `web/js/ode.js` — ODE dynamics panel
    - Display ODE dynamics plot (CycB, CDK1, APC over time)
    - Display transition probability matrix heatmap
    - Show "run pipeline first" message if no results
    - _Requirements: 7.1, 7.2, 7.6_

  - [ ] 10.3 Implement `web/js/hmm.js` — HMM comparison panel
    - Display ground truth, CNN-only, and HMM-corrected as three aligned horizontal tracks
    - Display CNN-only accuracy and HMM-corrected accuracy as percentages (1 decimal place) with difference
    - Display "Conceptual Demonstration" heading
    - Display disclaimer: "Demonstration: simulated time-lapse sequence (not real sequential predictions)"
    - Show "run pipeline first" message if no results
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 2.2, 2.3_

  - [ ] 10.4 Implement `web/js/anomaly.js` — anomaly detection panel
    - Display anomaly timeline visualization image
    - List each anomaly: severity, checkpoint type, biological interpretation
    - Show "no anomalies detected" message if list is empty
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 10.5 Implement `web/js/population.js` — population analysis panel
    - Display phase distribution chart (pie or bar chart with percentages)
    - Display mitotic index, growth fraction, doubling time, proliferation status
    - Display contextual limitation disclaimer from API response (visible without interaction)
    - _Requirements: 8.4, 8.5, 3.2_

  - [ ] 10.6 Implement `web/js/upload.js` — single image upload panel
    - File upload interface accepting PNG, JPG, TIF (max 10 MB)
    - Client-side validation: reject invalid format/size with inline error
    - Show loading indicator during classification
    - Display predicted phase with confidence percentage (1 decimal place)
    - Display bar chart of all 7 phase confidence scores
    - Display Grad-CAM heatmap overlay on uploaded image
    - Show "run pipeline first" message if no model trained
    - Show error message if classification fails
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [ ] 11. Checkpoint — Verify frontend integration
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Testing — Unit tests
  - [ ] 12.1 Create test directory structure and fixtures
    - Create `tests/` directory with `conftest.py`, `test_loader.py`, `test_model.py`, `test_ode.py`, `test_hmm.py`, `test_checkpoint.py`, `test_population.py`, `test_classifier.py`
    - Create synthetic fixtures: small dataset (5 images/phase), random tensors, known matrices
    - Configure pytest with 120-second timeout for unit+integration tests
    - _Requirements: 10.2, 10.5_

  - [ ] 12.2 Write unit tests for data loading (`tests/test_loader.py`)
    - Test stratified split preserves per-class ratios within 5% tolerance
    - Test Ground_truth.lst parsing and validation
    - Test empty-directory error raising
    - _Requirements: 10.1_

  - [ ] 12.3 Write unit tests for model and training (`tests/test_model.py`)
    - Test CNN forward pass output shape is `[batch_size, 7]`
    - Test softmax output sums to 1.0
    - Test class weight computation matches inverse-frequency formula
    - _Requirements: 10.1_

  - [ ] 12.4 Write unit tests for ODE, HMM, and checkpoint (`tests/test_ode.py`, `tests/test_hmm.py`, `tests/test_checkpoint.py`)
    - Test ODE solver output shape and value ranges (non-negative, ≤ 2.0)
    - Test HMM Viterbi decoding on known sequence produces expected path
    - Test checkpoint detector flags known invalid transitions
    - _Requirements: 10.1_

  - [ ] 12.5 Write unit tests for population analysis and classifier service (`tests/test_population.py`, `tests/test_classifier.py`)
    - Test disclaimer field is present in population output
    - Test < 2 predictions guard returns insufficient-data message
    - Test file validation logic (format + size acceptance/rejection)
    - _Requirements: 10.1_

- [ ] 13. Testing — Property-based tests
  - [ ] 13.1 Write property test for stratified split class proportions
    - **Property 1: Stratified split preserves class proportions**
    - Generate random datasets (20–500 samples, varying class distributions across 7 phases)
    - Split and assert per-class proportion tolerance ≤ 5%
    - Minimum 100 iterations with Hypothesis
    - **Validates: Requirements 1.2**

  - [ ] 13.2 Write property test for inverse-frequency class weight computation
    - **Property 2: Inverse-frequency class weight computation**
    - Generate random class count arrays (7 elements, each 1–1000)
    - Compute weights and assert `weight_i == total / (num_classes * count_i)`
    - Assert weight vector length equals number of classes
    - Minimum 100 iterations with Hypothesis
    - **Validates: Requirements 1.3**

  - [ ] 13.3 Write property test for ground truth label validation
    - **Property 3: Ground truth label validation**
    - Generate random ground truth mappings and directory structures
    - Run validation, assert correct flagging of mismatches and non-flagging of matches
    - Minimum 100 iterations with Hypothesis
    - **Validates: Requirements 1.6**

  - [ ] 13.4 Write property test for image upload validation
    - **Property 4: Image upload validation rejects invalid inputs**
    - Generate random files with various extensions and sizes (valid and invalid)
    - Assert invalid files (wrong format or > 10 MB) are rejected with correct error
    - Assert valid files (PNG/JPG/TIF, ≤ 10 MB) are accepted
    - Minimum 100 iterations with Hypothesis
    - **Validates: Requirements 4.5, 9.6**

- [ ] 14. Testing — Integration tests
  - [ ] 14.1 Write API integration tests (`tests/test_api_integration.py`)
    - Test all endpoints from Requirement 4 for correct HTTP status codes (200, 400, 404, 409)
    - Validate response body structure against Pydantic schemas
    - Verify CORS headers on cross-origin requests
    - Test pipeline run → progress polling → results retrieval flow
    - Use FastAPI `TestClient` for synchronous testing
    - _Requirements: 10.3_

- [ ] 15. Testing — End-to-end test
  - [ ] 15.1 Write end-to-end test (`tests/e2e/test_full_pipeline.py`)
    - Run full pipeline on ≤ 10 images per class
    - Verify all output artifacts: model file, plot images, JSON results are non-empty files
    - Separate from unit/integration tests (excluded from 120-second budget)
    - _Requirements: 10.4_

- [ ] 16. Documentation
  - [ ] 16.1 Create `README.md` at repository root
    - Project overview and architecture diagram (text-based)
    - Installation instructions with runnable example commands
    - Usage instructions for CLI and web interface
    - Dataset attribution (BBBC048)
    - Explicitly state which components are fully functional vs conceptual demonstrations
    - _Requirements: 11.1, 11.4_

  - [ ] 16.2 Create `ARCHITECTURE.md`
    - Explain 3-layer pipeline design
    - Describe biological reasoning behind each layer
    - Explain how ODE model connects to HMM
    - Include known limitations and future work section
    - _Requirements: 11.2, 11.5_

  - [ ] 16.3 Add inline code comments to ODE, HMM, and checkpoint modules
    - Add comments explaining biological significance of ODE constants
    - Add comments explaining checkpoint thresholds and phase durations
    - Add comments explaining HMM transition/emission matrix derivation
    - _Requirements: 11.3_

- [ ] 17. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The frontend uses vanilla HTML/CSS/JS (no framework) per architectural decision
- Python is the implementation language for all backend code (FastAPI, pytest, Hypothesis)
- All tests (unit + integration) must complete within 120 seconds excluding E2E tests

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "5.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "2.1", "3.1", "5.2"] },
    { "id": 2, "tasks": ["6.1", "6.2", "8.1"] },
    { "id": 3, "tasks": ["6.3", "6.4", "6.5", "6.6", "8.2"] },
    { "id": 4, "tasks": ["9.1", "9.2"] },
    { "id": 5, "tasks": ["10.1", "10.2", "10.3", "10.4", "10.5", "10.6"] },
    { "id": 6, "tasks": ["12.1"] },
    { "id": 7, "tasks": ["12.2", "12.3", "12.4", "12.5"] },
    { "id": 8, "tasks": ["13.1", "13.2", "13.3", "13.4", "14.1"] },
    { "id": 9, "tasks": ["15.1"] },
    { "id": 10, "tasks": ["16.1", "16.2", "16.3"] }
  ]
}
```
