# JavaScript Architecture Visualization

## Overview

This document shows the architecture of the MANTAX Face Recognition app's JavaScript modules and how they interact with user workflows.

---

## Old Architecture (Monolithic app.js)

```mermaid
graph TD
    subgraph "Old: Single File (app.js - 3,429 lines)"
        A["app.js<br/>98 Functions<br/>All Logic"]
        
        A1["API Calls"]
        A2["State Management"]
        A3["UI Updates"]
        A4["Event Handlers"]
        A5["Workflows"]
        A6["Visualizations"]
        
        A --> A1
        A --> A2
        A --> A3
        A --> A4
        A --> A5
        A --> A6
    end
    
    User --> A
    
    style A fill:#ff9999,stroke:#333,stroke-width:2px
    style A1 fill:#ffcccc,stroke:#333
    style A2 fill:#ffcccc,stroke:#333
    style A3 fill:#ffcccc,stroke:#333
    style A4 fill:#ffcccc,stroke:#333
    style A5 fill:#ffcccc,stroke:#333
    style A6 fill:#ffcccc,stroke:#333
```

### Problems with Old Architecture:
- Single file with 3,429 lines
- 98 functions mixed together
- No separation of concerns
- Hard to maintain and test
- All state scattered throughout

---

## New Architecture (Modular)

```mermaid
graph TD
    subgraph "Frontend (Browser)"
        HTML["index.html<br/>Event Handlers"]
        
        subgraph "New Modular JS Files"
            INIT["07-init.js<br/>Bootstrap & Exports<br/>101 Functions"]
            API["01-api.js<br/>API Client<br/>31 Endpoints"]
            STATE["02-state.js<br/>State Container"]
            WORKFLOWS["03-workflows.js<br/>User Actions"]
            COMPARE["04-compare.js<br/>Comparison Logic"]
            UI["05-ui.js<br/>Toast, Loading<br/>Terminal"]
            VIZ["06-viz.js<br/>Visualizations"]
        end
        
        HTML --> INIT
        INIT --> API
        INIT --> STATE
        INIT --> WORKFLOWS
        INIT --> COMPARE
        INIT --> UI
        INIT --> VIZ
    end
    
    API --> |"HTTP Requests"| BACKEND["Flask API Server<br/>api_server.py"]
    BACKEND --> |"JSON Responses"| API
    
    STATE --> |"Reads/Writes"| STATE
    WORKFLOWS --> |"Uses"| API
    WORKFLOWS --> |"Uses"| STATE
    WORKFLOWS --> |"Uses"| UI
    COMPARE --> |"Uses"| API
    COMPARE --> |"Uses"| VIZ
    UI --> |"Updates"| HTML
    
    style INIT fill:#99ff99,stroke:#333,stroke-width:2px
    style API fill:#99ccff,stroke:#333
    style STATE fill:#ffcc99,stroke:#333
    style WORKFLOWS fill:#ff99cc,stroke:#333
    style COMPARE fill:#cc99ff,stroke:#333
    style UI fill:#ffff99,stroke:#333
    style VIZ fill:#99ffff,stroke:#333
    style HTML fill:#ffcccc,stroke:#333
    style BACKEND fill:#ccccff,stroke:#333,stroke-width:2px
```

---

## Module Dependencies

```mermaid
graph TD
    subgraph "Dependency Direction (High → Low)"
        HIGH["07-init.js<br/>High Level<br/>Bootstrap & Exports"]
        MID1["03-workflows.js<br/>Mid Level<br/>User Actions"]
        MID2["04-compare.js<br/>Mid Level<br/>Comparison Logic"]
        LOW1["01-api.js<br/>Low Level<br/>API Client"]
        LOW2["02-state.js<br/>Low Level<br/>State Container"]
        LOW3["05-ui.js<br/>Low Level<br/>UI Components"]
        LOW4["06-viz.js<br/>Low Level<br/>Visualizations"]
    end
    
    HIGH --> MID1
    HIGH --> MID2
    MID1 --> LOW1
    MID1 --> LOW2
    MID1 --> LOW3
    MID1 --> LOW4
    MID2 --> LOW1
    MID2 --> LOW4
    MID2 --> LOW3
    
    style HIGH fill:#99ff99,stroke:#333
    style MID1 fill:#ff99cc,stroke:#333
    style MID2 fill:#cc99ff,stroke:#333
    style LOW1 fill:#99ccff,stroke:#333
    style LOW2 fill:#ffcc99,stroke:#333
    style LOW3 fill:#ffff99,stroke:#333
    style LOW4 fill:#99ffff,stroke:#333
```

---

## Workflow to Module Mapping

### Workflow 1: Upload → Match

```mermaid
sequenceDiagram
    participant User
    participant HTML
    participant INIT as 07-init.js
    participant WORKFLOWS as 03-workflows.js
    participant API as 01-api.js
    participant STATE as 02-state.js
    participant UI as 05-ui.js
    participant BACKEND as Flask API

    User->>HTML: Click "Choose Photo"
    HTML->>INIT: selectImage()
    INIT->>WORKFLOWS: selectImage()
    WORKFLOWS->>WORKFLOWS: handleImageSelect()
    WORKFLOWS->>API: POST /api/detect
    API->>BACKEND: Detect faces
    BACKEND-->>API: face data
    API-->>WORKFLOWS: detection result
    WORKFLOWS->>STATE: Store currentImage
    WORKFLOWS->>UI: showToast("Face detected")
    
    User->>HTML: Click "Find Faces"
    HTML->>INIT: detectFaces()
    INIT->>API: POST /api/extract
    API-->>WORKFLOWS: embedding data
    WORKFLOWS->>STATE: Store embedding
    
    User->>HTML: Click "Create Signature"
    HTML->>INIT: extractFeatures()
    INIT->>API: POST /api/extract
    API-->>WORKFLOWS: features
    WORKFLOWS->>STATE: Store features
    
    User->>HTML: Click "Compare"
    HTML->>INIT: compareFaces()
    INIT->>WORKFLOWS: compareFaces()
    WORKFLOWS->>API: POST /api/compare
    API-->>WORKFLOWS: comparison results
    WORKFLOWS->>UI: Display results
```

### Workflow 2: Webcam → Match

```mermaid
sequenceDiagram
    participant User
    participant HTML
    participant INIT as 07-init.js
    participant WORKFLOWS as 03-workflows.js
    participant UI as 05-ui.js

    User->>HTML: Click "Start Webcam"
    HTML->>INIT: startWebcam()
    INIT->>WORKFLOWS: startWebcam()
    WORKFLOWS->>UI: Show video stream
    
    User->>HTML: Click "Capture"
    HTML->>INIT: captureWebcam()
    INIT->>WORKFLOWS: captureWebcam()
    WORKFLOWS->>UI: Show captured frame
    WORKFLOWS->>UI: Show "Use for Matching" button
    
    User->>HTML: Click "Use for Matching"
    HTML->>INIT: useForMatching()
    INIT->>WORKFLOWS: useForMatching()
    WORKFLOWS->>WORKFLOWS: Continue to detection workflow
```

### Workflow 3: Library Management

```mermaid
sequenceDiagram
    participant User
    participant HTML
    participant INIT as 07-init.js
    participant API as 01-api.js
    participant COMPARE as 04-compare.js

    User->>HTML: Click "Find Matches"
    HTML->>INIT: matchWithLibraryImage()
    INIT->>COMPARE: matchWithLibraryImage()
    COMPARE->>API: POST /api/library/match
    API-->>COMPARE: matches
    COMPARE->>COMPARE: Display results
    
    User->>HTML: Click "+ Add Person"
    HTML->>INIT: showLibraryModal()
    INIT->>UI: Show modal
    
    User->>HTML: Fill form, Click "Save"
    HTML->>INIT: saveToLibrary()
    INIT->>API: POST /api/library/person
    API-->>INIT: saved person
    INIT->>UI: showToast("Saved")
```

---

## Visualization Workflow

```mermaid
flowchart LR
    subgraph User_Action
        A[User clicks viz tab]
    end
    
    subgraph HTML
        B[Button with data-viz attr]
    end
    
    subgraph "06-viz.js"
        C[showVisualization]
        D[fetchVizData]
        E[renderVizImage]
        F[renderVizData]
    end
    
    subgraph "01-api.js"
        G[callAPI]
    end
    
    subgraph "05-ui.js"
        H[showLoading]
        I[hideLoading]
    end
    
    A --> B --> INIT --> C
    C --> H
    C --> D
    D --> G
    G --> |JSON| D
    D --> I
    D --> E
    D --> F
    
    style C fill:#cc99ff,stroke:#333
    style D fill:#cc99ff,stroke:#333
    style E fill:#cc99ff,stroke:#333
    style F fill:#cc99ff,stroke:#333
```

---

## State Management (02-state.js)

```mermaid
classDiagram
    class AppState {
        +currentImage: string
        +currentEmbedding: array
        +currentFace: object
        +references: array
        +libraryPersons: array
        +comparisonResults: array
        +visualizationData: object
        +webcamStream: object
        +isLoading: boolean
        
        +get(key)
        +set(key, value)
        +clear()
    }
    
    class StateManager {
        +subscribe(callback)
        +unsubscribe(callback)
        +notify()
    }
    
    AppState --> StateManager
```

---

## File Comparison

| Aspect | Old (app.js) | New (Modular) |
|--------|-------------|----------------|
| **Total Lines** | 3,429 | ~2,000 (7 files) |
| **Functions per file** | 98 | 10-15 average |
| **Separation** | None | By responsibility |
| **Testability** | Hard | Easy |
| **Maintainability** | Difficult | Good |
| **Reusability** | Low | High |

---

## Module Responsibility Summary

```mermaid
graph TD
    subgraph "01-api.js"
        A1["detectFaces()"]
        A2["extractFeatures()"]
        A3["addReference()"]
        A4["compareFaces()"]
        A5["getVisualization()"]
        A6["libraryMatch()"]
    end
    
    subgraph "02-state.js"
        S1["getState()"]
        S2["setState()"]
        S3["subscribe()"]
    end
    
    subgraph "03-workflows.js"
        W1["selectImage()"]
        W2["handleImageSelect()"]
        W3["startWebcam()"]
        W4["captureWebcam()"]
        W5["saveToLibrary()"]
    end
    
    subgraph "04-compare.js"
        C1["compareWithReferences()"]
        C2["compareWithLibrary()"]
        C3["findMatches()"]
    end
    
    subgraph "05-ui.js"
        U1["showToast()"]
        U2["showLoading()"]
        U3["logToTerminal()"]
        U4["updateReferenceList()"]
    end
    
    subgraph "06-viz.js"
        V1["showVisualization()"]
        V2["renderEmbedding()"]
        V3["renderSimilarity()"]
    end
    
    subgraph "07-init.js"
        I1["init()"]
        I2["exposeFunctions()"]
    end
    
    style A1 fill:#99ccff,stroke:#333
    style S1 fill:#ffcc99,stroke:#333
    style W1 fill:#ff99cc,stroke:#333
    style C1 fill:#cc99ff,stroke:#333
    style U1 fill:#ffff99,stroke:#333
    style V1 fill:#99ffff,stroke:#333
    style I1 fill:#99ff99,stroke:#333
```

---

## User Flow → Module Routing

```mermaid
flowchart TB
    subgraph Input
        UPLOAD["Upload Image"]
        WEBCAM["Webcam"]
        LIBRARY["Library"]
    end
    
    subgraph Processing
        DETECT["Detect Faces"]
        EXTRACT["Extract Features"]
        COMPARE["Compare"]
    end
    
    subgraph Output
        VIZ["Visualizations"]
        RESULTS["Results Display"]
        SAVE["Save to Library"]
    end
    
    UPLOAD --> DETECT
    WEBCAM --> DETECT
    DETECT --> EXTRACT
    EXTRACT --> COMPARE
    COMPARE --> VIZ
    COMPARE --> RESULTS
    COMPARE --> SAVE
    LIBRARY --> COMPARE
    
    style UPLOAD fill:#ffcccc
    style WEBCAM fill:#ffcccc
    style LIBRARY fill:#ffcccc
    style DETECT fill:#ffcc99
    style EXTRACT fill:#ffcc99
    style COMPARE fill:#cc99ff
    style VIZ fill:#99ffff
    style RESULTS fill:#99ff99
    style SAVE fill:#99ccff
```

---

## Integration with HTML

```mermaid
graph LR
    subgraph HTML
        BTN["button onclick='functionName()'"]
    end
    
    subgraph "07-init.js"
        EXPORT["window.functionName = function"]
    end
    
    subgraph "03-workflows.js"
        HANDLER["function functionName()"]
    end
    
    subgraph "01-api.js"
        API["fetch() calls"]
    end
    
    subgraph Backend
        FLASK["Flask API Server"]
    end
    
    BTN --> EXPORT --> HANDLER --> API --> FLASK
    
    style BTN fill:#ffcccc
    style EXPORT fill:#99ff99
    style HANDLER fill:#ff99cc
    style API fill:#99ccff
    style FLASK fill:#ccccff
```

---

## Summary

The new modular architecture provides:

1. **Single Responsibility**: Each file handles one concern
2. **Dependency Direction**: High-level → Mid-level → Low-level
3. **Testability**: Each module can be tested independently
4. **Maintainability**: Easier to find and fix bugs
5. **Reusability**: Functions can be reused across workflows
6. **Scalability**: New features can be added without touching existing code

The 7 modular files replace the single 3,429-line app.js while maintaining all 98 functions and adding better organization.
