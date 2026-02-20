# JavaScript Workflow Analysis

## Overview
This document analyzes all workflows in the Face Recognition app to ensure they are independent and can work from any entry point to any endpoint.

## Test Results (February 20, 2026)
- **Total Tests**: 24
- **Passed**: 24
- **Failed**: 0
- **Key Finding**: Comparison result shows Match: WorkflowTestRef, Score: 27%, Viz types: 12

## Workflows Diagram (Mermaid)

```mermaid
flowchart TD
    subgraph Upload_Workflow
        A1[User clicks Choose Photo] --> A2[handleImageSelect]
        A2 --> A3[File validation]
        A3 --> A4[currentImage = base64]
        A4 --> A5[Update UI previews]
        A5 --> A6[resetSteps]
        A6 --> A7[Auto: detectFaces]
        A7 --> A8[Auto: extractFeatures]
        A8 --> A9[Auto: pre-cache all viz]
    end

    subgraph Detection_Workflow
        B1[detectFaces API call] --> B2[POST /api/detect]
        B2 --> B3[Store faces in currentFaceThumbnails]
        B3 --> B4[Update gallery UI]
        B4 --> B5[Enable extractBtn]
    end

    subgraph Extraction_Workflow
        C1[extractFeatures API call] --> C2[POST /api/extract]
        C2 --> C3[Store embedding in currentQueryEmbedding]
        C3 --> C4[Store viz data in visualizationData]
        C4 --> C5[Pre-cache all viz types]
        C5 --> C6[Enable compareBtn]
    end

    subgraph Compare_Workflow
        D1[compareFaces] --> D2{references > 0?}
        D2 -->|No| D3[Show error: Add reference]
        D2 -->|Yes| D4[POST /api/compare]
        D4 --> D5[Show comparisonResult]
        D5 --> D6[Show match status/scores]
        D5 --> D7[Show query/ref images]
    end

    subgraph Reference_Workflow
        E1[addReference] --> E2[refInput.click]
        E2 --> E3[handleReferenceSelect]
        E3 --> E4[saveReference API]
        E4 --> E5[references.push]
        E5 --> E6[updateReferenceList]
        E6 --> D6
    end

    subgraph Library_Workflow
        F1[Library Upload] --> F2[handleLibraryUpload]
        F2 --> F3[saveToLibrary API]
        F3 --> F4[renderLibraryGrid]
    end

    subgraph Webcam_Workflow
        G1[startWebcam] --> G2[navigator.mediaDevices.getUserMedia]
        G2 --> G3[initFaceMesh]
        G3 --> G4[processWebcamFrame]
        G4 --> G5[drawMesh overlay]
    end

    subgraph Clear_Workflow
        H1[clearAllCache] --> H2[POST /api/clear]
        H2 --> H3[currentImage = null]
        H3 --> H4[currentFaceThumbnails = []]
        H4 --> H5[currentQueryEmbedding = null]
        H5 --> H6[references = []]
        H5 --> H7[visualizationData = {}]
        H7 --> H8[Reset all UI elements]
    end

    subgraph Visualization_Workflow
        I1[Click viz tab] --> I2{Already cached?}
        I2 -->|Yes| I3[Show from cache]
        I2 -->|No| I4[Fetch from API]
        I4 --> I5[Store in cache]
        I5 --> I3
    end
```

## Key Entry Points

| Entry Point | Function | Required State |
|-------------|----------|----------------|
| Choose Photo | `handleImageSelect` | None |
| Upload Ref | `handleReferenceSelect` | None |
| Find Faces | `detectFaces` | `currentImage` |
| Create Signature | `extractFeatures` | `currentFaceThumbnails.length > 0` |
| Compare | `compareFaces` | `currentQueryEmbedding !== null` AND `references.length > 0` |
| Library Upload | `handleLibraryUpload` | None |
| Webcam | `startWebcam` | None |
| Clear | `clearAllCache` | None |

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/detect` | POST | Face detection |
| `/api/extract` | POST | Embedding extraction |
| `/api/add-reference` | POST | Add reference |
| `/api/references` | GET | List references |
| `/api/compare` | POST | Compare embeddings |
| `/api/visualizations/<type>` | GET | Get visualization |
| `/api/clear` | POST | Clear session |

## Visualization Types (12+)

1. detection
2. extraction
3. preprocessing
4. landmarks
5. mesh3d
6. alignment
7. saliency
8. activations
9. features
10. multiscale
11. confidence
12. eyewear
13. embedding
14. similarity
15. robustness
16. biometric
17. asymmetry
18. texture
19. normalized

## Issues Found

### Issue 1: Auto-run Chaining
- `handleImageSelect` auto-runs detection + extraction
- This breaks workflow independence - user can't upload without auto-running

### Issue 2: State Dependencies
- `detectFaces` checks `currentImage` but doesn't show clear error if missing
- `extractFeatures` checks `currentFaceThumbnails` but silently returns
- `compareFaces` checks both but errors could be clearer

### Issue 3: Missing State Checks
- No guard at start of visualization functions
- Compare button enabled state not properly managed

### Issue 4: Null Element References
- Many places assume DOM elements exist
- Need better null checks throughout

## Recommendations

1. Add explicit state validation at each entry point
2. Make auto-run optional or configurable  
3. Add visual indicators for required state
4. Ensure all functions can run independently

## Comparison Result Structure

```json
{
  "success": true,
  "best_match": {
    "name": "WorkflowTestRef",
    "final_score": 0.27,
    "match_label": "No Match",
    "status": "no_match",
    "arcface_similarity": null,
    "facenet_similarity": null,
    "activation_similarity": null,
    "normalized_similarity": null,
    "multi_pose_score": null,
    "lbp_similarity": null,
    "asymmetry_similarity": null,
    "thumbnail": "base64..."
  },
  "similarity_viz": "base64...",
  "similarity_data": {...}
}
```
