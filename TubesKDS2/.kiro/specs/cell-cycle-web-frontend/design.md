# Design Document: Cell Cycle Web Frontend

## Overview

This design describes the web-based interface and backend API that wraps the existing Cell Cycle Intelligence System — a 3-layer Python ML pipeline for classifying cell cycle phases from BBBC048 fluorescence microscopy images. The system adds:

1. **Backend pipeline fixes** — correcting data loading honesty, HMM transparency, and population analysis context
2. **REST API** — a FastAPI server exposing all pipeline operations as HTTP endpoints
3. **Web frontend** — a single-page application (HTML/CSS/JS) providing dashboard, visualization panels, and single-image classification
4. **Testing** — unit, integration, and end-to-end test suites
5. **Documentation** — README, architecture docs, and inline comments

The design preserves the existing pipeline structure while adding a web layer on top and fixing known issues in the underlying code.

## Architecture

```mermaid
graph TB
    subgraph Frontend["Web Frontend (HTML/CSS/JS)"]
        Dashboard[Dashboard Page]
        NavBar[Navigation Bar]
        PipelinePanel[Pipeline Control]
        ClassPanel[Classification Results]
        ODEPanel[ODE Dynamics Panel]
        HMMPanel[HMM Comparison Panel]
        AnomalyPanel[Anomaly Detection Panel]
        PopPanel[Population Analysis Panel]
        UploadPanel[Single Image Upload]
    end

    subgraph API["Backend API (FastAPI)"]
        PipelineEndpoint[POST /api/pipeline/run]
        ProgressEndpoint[GET /api/pipeline/progress]
        ResultsEndpoint[GET /api/results]
        PlotsEndpoint[GET /api/plots/{name}]
        ClassifyEndpoint[POST /api/classify]
        HealthEndpoint[GET /api/health]
    end

    subgraph Pipeline["ML Pipeline (Python)"]
        DataLoader[bbbc048_loader.py]
        CNN[CellCycleCNN - ResNet-18]
        ODE[Cyclin-CDK ODE Solver]
        HMM[Biological HMM]
        Checkpoint[Checkpoint Detector]
        Population[Population Analyzer]
        GradCAM[Grad-CAM Explainability]
        Viz[Visualization Module]
    end

    subgraph Storage["File System"]
        Data[(BBBC048 Dataset)]
        Models[(Trained Models)]
        Plots[(Generated Plots)]
        Results[(JSON Results)]
    end

    Frontend -->|HTTP/REST| API
    API -->|Invokes| Pipeline
    Pipeline -->|Reads| Data
    Pipeline -->|Writes| Models
    Pipeline -->|Writes| Plots
    Pipeline -->|Writes| Results
    API -->|Serves| Plots
    API -->|Serves| Results
```

### Key Architectural Decisions

1. **FastAPI over Flask**: FastAPI provides async support (needed for long-running pipeline jobs), automatic OpenAPI docs, built-in validation with Pydantic, and better performance. The pipeline training is CPU/GPU-bound, so async background tasks prevent blocking the API.

2. **Vanilla HTML/CSS/JS frontend**: The project is a university assignment; a framework like React would add unnecessary complexity. A single `index.html` with modular JS files keeps things simple and dependency-free on the frontend.

3. **File-based result storage**: Pipeline outputs (plots, JSON results) are written to disk. The API serves them directly. No database is needed — the system processes one pipeline run at a time.

4. **Background task for pipeline execution**: Training takes minutes. The API starts it in a background thread and exposes a progress polling endpoint. The frontend polls every 2 seconds.

## Components and Interfaces

### Backend API (FastAPI)

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/api/health` | GET | Health check | `{"status": "ok"}` |
| `/api/pipeline/run` | POST | Start full pipeline execution | `{"job_id": str, "status": "started"}` |
| `/api/pipeline/progress` | GET | Get current training progress | `{"status": str, "epoch": int, "total_epochs": int, "loss": float, "accuracy": float}` |
| `/api/results` | GET | Get latest pipeline results | JSON with confusion matrix, classification report, ODE data, HMM data, anomalies, population summary |
| `/api/plots/{name}` | GET | Get a generated plot image | PNG image |
| `/api/classify` | POST | Classify a single uploaded image | `{"phase": str, "confidence": float, "all_scores": dict, "gradcam_url": str}` |

### Pipeline Module Changes

| Module | Change | Rationale |
|--------|--------|-----------|
| `synthetic_data.py` → `bbbc048_loader.py` | Rename file | Requirement 1.1 — honest naming |
| `dataset.py` | Add stratified splitting, class weighting | Requirements 1.2, 1.3 |
| `model.py` | Move numpy import to top, set NUM_EPOCHS=20 | Requirements 1.4, 1.5 |
| `hmm.py` | Add transparency docstrings/comments | Requirement 2.1 |
| `population_analysis.py` | Add disclaimer field, minimum-count guard | Requirements 3.1, 3.3 |

### Frontend Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Main Layout | `index.html` | Page structure, navigation, panel containers |
| Styles | `styles.css` | Dark-themed responsive layout |
| App Controller | `app.js` | Navigation state, API communication, polling |
| Pipeline Panel | `pipeline.js` | Run button, progress display |
| Results Panel | `results.js` | Confusion matrix, metrics table, training curves |
| ODE Panel | `ode.js` | ODE dynamics plot, transition matrix heatmap |
| HMM Panel | `hmm.js` | HMM comparison tracks, disclaimer display |
| Anomaly Panel | `anomaly.js` | Anomaly timeline, anomaly list |
| Population Panel | `population.js` | Phase distribution chart, proliferation metrics |
| Upload Panel | `upload.js` | File upload, classification result, Grad-CAM display |

### API Server Structure

```
api/
├── __init__.py
├── server.py          # FastAPI app, CORS, static file serving
├── routes/
│   ├── __init__.py
│   ├── pipeline.py    # /api/pipeline/* endpoints
│   ├── results.py     # /api/results, /api/plots/*
│   └── classify.py    # /api/classify endpoint
├── services/
│   ├── __init__.py
│   ├── pipeline_runner.py  # Background pipeline execution
│   └── classifier.py      # Single-image classification logic
└── models/
    ├── __init__.py
    └── schemas.py     # Pydantic request/response models
```

### Frontend Structure

```
web/
├── index.html
├── css/
│   └── styles.css
├── js/
│   ├── app.js
│   ├── pipeline.js
│   ├── results.js
│   ├── ode.js
│   ├── hmm.js
│   ├── anomaly.js
│   ├── population.js
│   └── upload.js
└── assets/
    └── (icons, placeholder images)
```

## Data Models

### API Response Schemas (Pydantic)

```python
from pydantic import BaseModel
from typing import Optional, List, Dict
from enum import Enum

class JobStatus(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"

class PipelineRunResponse(BaseModel):
    job_id: str
    status: JobStatus

class ProgressResponse(BaseModel):
    status: JobStatus
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    last_run_status: Optional[JobStatus] = None
    error_message: Optional[str] = None

class ClassificationResult(BaseModel):
    phase: str
    confidence: float
    all_scores: Dict[str, float]  # phase_name -> confidence
    gradcam_url: str

class AnomalyItem(BaseModel):
    severity: str
    checkpoint: str
    phase: str
    frame_start: int
    frame_end: int
    bio_interpretation: str

class PopulationResult(BaseModel):
    phase_distribution: Dict[str, float]
    mitotic_index: float
    growth_fraction: float
    doubling_time_hours: float
    status: str
    clinical_significance: str
    disclaimer: str
    total_cells_analyzed: int

class HMMResult(BaseModel):
    ground_truth: List[str]
    cnn_predictions: List[str]
    hmm_corrected: List[str]
    cnn_accuracy: float
    hmm_accuracy: float
    is_demonstration: bool  # Always True — Requirement 2.4
    disclaimer: str

class PipelineResults(BaseModel):
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Dict[str, float]]
    test_accuracy: float
    ode_data: Dict  # t, CycB, CDK1, APC arrays
    hmm_result: HMMResult
    anomalies: List[AnomalyItem]
    population: PopulationResult
    available_plots: List[str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
```

### Pipeline State (In-Memory)

```python
@dataclass
class PipelineState:
    status: JobStatus = JobStatus.idle
    job_id: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 20
    current_loss: float = 0.0
    current_accuracy: float = 0.0
    error_message: Optional[str] = None
    results: Optional[dict] = None  # Cached latest results
    model: Optional[Any] = None     # Loaded trained model
```

### File-Based Results Storage

Pipeline results are persisted as JSON in `output/results/`:

```
output/
├── models/
│   └── best_model.pth
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── ode_dynamics.png
│   ├── transition_matrix.png
│   ├── hmm_comparison.png
│   ├── anomaly_timeline.png
│   ├── population_dashboard.png
│   └── gradcam/
│       └── upload_gradcam.png
└── results/
    └── pipeline_results.json
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Stratified split preserves class proportions

*For any* dataset with an arbitrary class distribution across 7 phases, when the Data_Loader performs a stratified split into train (70%), validation (15%), and test (15%) subsets, each subset's per-class proportion SHALL differ from the overall dataset's per-class proportion by no more than 5 percentage points.

**Validates: Requirements 1.2**

### Property 2: Inverse-frequency class weight computation

*For any* set of class counts (where each count is at least 1), the computed class weight for class `i` SHALL equal `total_samples / (num_classes * count_i)`, and the resulting weight vector SHALL have length equal to the number of classes.

**Validates: Requirements 1.3**

### Property 3: Ground truth label validation

*For any* Ground_truth.lst file mapping image filenames to phase labels, and a corresponding directory structure with images in phase subdirectories, the Data_Loader validation SHALL flag every image whose Ground_truth.lst label differs from its directory assignment, and SHALL not flag any image whose labels match.

**Validates: Requirements 1.6**

### Property 4: Image upload validation rejects invalid inputs

*For any* file that either (a) has a format other than PNG, JPG, or TIF, or (b) exceeds 10 MB in size, the image validation logic SHALL reject the file and return an error message indicating the accepted formats and size limit. Conversely, *for any* valid PNG, JPG, or TIF file not exceeding 10 MB, the validation SHALL accept the file.

**Validates: Requirements 4.5, 9.6**

## Error Handling

### Backend API Error Strategy

| Scenario | HTTP Status | Response Body |
|----------|-------------|---------------|
| Pipeline already running | 409 Conflict | `{"error": "Pipeline execution already in progress"}` |
| No trained model exists | 404 Not Found | `{"error": "No trained model available. Run the pipeline first."}` |
| Invalid image format/size | 400 Bad Request | `{"error": "Invalid file. Accepted: PNG, JPG, TIF up to 10 MB."}` |
| Pipeline execution fails | 500 Internal Error | `{"error": "Pipeline failed", "detail": "<exception message>"}` |
| Results not available | 404 Not Found | `{"error": "No results available. Run the pipeline first."}` |
| Plot not found | 404 Not Found | `{"error": "Plot not found: <name>"}` |
| Progress with no active run | 200 OK | `{"status": "idle", "last_run_status": "completed"/"failed"/null}` |

### Frontend Error Handling

- **Network errors**: Display "Cannot connect to server. Please ensure the backend is running." with a retry button.
- **API errors (4xx/5xx)**: Parse error message from response body and display in a dismissible alert banner within the relevant panel.
- **File validation errors**: Display inline error below the upload area before sending to the API.
- **Polling failures**: After 3 consecutive poll failures, stop polling and display a connection error with a manual refresh button.

### Pipeline Error Handling

- **Missing dataset**: Raise `FileNotFoundError` with message indicating which directory/file is missing.
- **Empty phase directory**: Raise `ValueError` listing which phase subdirectories have fewer than 1 image (Requirement 1.7).
- **Training failure (OOM, NaN loss)**: Catch exception, save partial state, report error through progress endpoint.
- **ODE solver failure**: Catch `scipy` integration errors, report with biological context.
- **Population analysis with < 2 predictions**: Return early with insufficient-data message (Requirement 3.3).

## Testing Strategy

### Test Framework

- **pytest** for all Python tests (unit, integration, E2E)
- **pytest-asyncio** for async API endpoint tests
- **httpx** (or `TestClient` from FastAPI) for API integration tests
- **Hypothesis** for property-based testing (Python PBT library)

### Unit Tests

| Module | Tests | Fixtures |
|--------|-------|----------|
| `bbbc048_loader.py` | Stratified split ratios, Ground_truth.lst parsing, empty-dir error | Synthetic small dataset (5 images/phase) |
| `dataset.py` | Class weight computation, transform application | Random tensors |
| `model.py` | Forward pass shape `[batch, 7]`, softmax sums to 1.0 | Random input tensors |
| `ode_model.py` | Output shape, concentrations non-negative and ≤ 2.0 | Default ODE params |
| `hmm.py` | Viterbi on known sequence produces expected path | Known transition/emission matrices |
| `checkpoint_detector.py` | Known invalid transitions produce expected anomaly flags | Constructed phase sequences |
| `population_analysis.py` | Disclaimer field present, < 2 predictions guard | Small prediction arrays |
| `api/services/classifier.py` | File validation logic (format + size) | Synthetic file objects |

### Property-Based Tests (Hypothesis)

Each property test runs a minimum of 100 iterations:

- **Property 1**: Generate random datasets (varying sizes 20–500, varying class distributions), split, assert per-class proportion tolerance ≤ 5%.
  - Tag: `Feature: cell-cycle-web-frontend, Property 1: Stratified split preserves class proportions`
- **Property 2**: Generate random class count arrays (7 elements, each 1–1000), compute weights, assert formula correctness.
  - Tag: `Feature: cell-cycle-web-frontend, Property 2: Inverse-frequency class weight computation`
- **Property 3**: Generate random ground truth mappings and directory structures, run validation, assert correct flagging.
  - Tag: `Feature: cell-cycle-web-frontend, Property 3: Ground truth label validation`
- **Property 4**: Generate random files with various extensions and sizes (valid and invalid), run validation, assert correct accept/reject.
  - Tag: `Feature: cell-cycle-web-frontend, Property 4: Image upload validation rejects invalid inputs`

### Integration Tests

- All API endpoints from Requirement 4 tested for correct HTTP status codes (200, 400, 404, 409)
- Response body structure validated against Pydantic schemas
- CORS headers verified on cross-origin requests
- Pipeline run → progress polling → results retrieval flow tested end-to-end

### End-to-End Test

- Runs full pipeline on ≤ 10 images per class
- Verifies all output artifacts (model file, plot images, JSON results) are produced as non-empty files
- Separated from unit/integration tests (excluded from 120-second budget)

### Test Execution

```bash
# Unit + integration tests (must complete within 120 seconds)
pytest tests/ -x --ignore=tests/e2e/ --timeout=120

# End-to-end test (separate, longer running)
pytest tests/e2e/ -v

# Property-based tests specifically
pytest tests/ -k "property" --hypothesis-show-statistics
```

