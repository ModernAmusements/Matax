# User Flows - MANTAX Face Recognition

## Overview

This document describes all user flows in the MANTAX Face Recognition application.

---

## Flow 1: Upload → Match (Existing)

```
Step 1: Choose Photo
    ↓
User uploads image
    ↓
Step 2: Find Faces
    ↓
AI detects face(s)
    ↓
Step 3: Extract Features
    ↓
AI extracts embeddings
    ↓
Step 4: Compare
    ↓
User adds reference(s)
    ↓
Compare → Show results
```

---

## Flow 2: Webcam → Match (Existing)

```
Step 1: Choose Photo
    ↓
User clicks "Start Webcam"
    ↓
Live webcam feed displays
    ↓
User clicks "Capture"
    ↓
[Use for Matching] button appears
    ↓
Click "Use for Matching"
    ↓
Continue to Step 2-4
```

---

## Flow 3: Upload → Save to Library (NEW)

```
Step 1: Choose Photo
    ↓
User uploads image
    ↓
(Skip to Step 6 - Reference Library)
    ↓
Click "+ Add from Upload"
    ↓
Select image file
    ↓
Modal appears:
  - Preview image
  - Enter person name *
  - Enter notes (optional)
    ↓
Click "Save"
    ↓
Person saved to library
```

---

## Flow 4: Webcam → Save to Library (NEW)

```
Step 1: Choose Photo
    ↓
User clicks "Start Webcam"
    ↓
Live webcam feed displays
    ↓
User clicks "Capture"
    ↓
[Save to Library] button appears
    ↓
Click "Save to Library"
    ↓
Modal appears:
  - Preview image (captured)
  - Enter person name *
  - Enter notes (optional)
    ↓
Click "Save"
    ↓
Person saved to library
```

---

## Flow 5: Match with Library (NEW)

```
Step 1-3: Choose → Detect → Extract
    ↓
User has query image ready
    ↓
Step 4: Compare
    ↓
Click "Compare with Library"
    ↓
System compares against all
    library persons
    ↓
Returns best match per person
    ↓
Display: Person name + score
```

---

## Flow 6: Manage Library (NEW)

```
Step 6: Reference Library
    ↓
View all saved persons
    ↓
- Click person to view details
- Click "+ Add from Upload" to add
- Click "+ Add from Webcam" to add
- Click "Delete" to remove
```

---

## Data Storage

### Reference Library
```
reference_library/
└── persons/
    └── {uuid}-{name}/
        ├── metadata.json    # name, notes, dates
        ├── images/
        │   └── {id}.jpg    # Compressed 80% JPEG
        └── embeddings.json # All extracted features
```

### Git Ignore
- `reference_library/` is in `.gitignore`
- Never committed to git

---

## Image Processing

When an image is saved to library:
1. Face detection
2. Face extraction (crop)
3. Embedding extraction (ArcFace + FaceNet)
4. Pose estimation (yaw/pitch/roll)
5. Quality metrics
6. LBP histogram
7. Asymmetry features
8. Image compression (JPEG 80%)
9. Save to JSON

---

## API Endpoints - Library

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/library` | List all persons |
| POST | `/api/library/person` | Add new person |
| GET | `/api/library/person/<id>` | Get person details |
| DELETE | `/api/library/person/<id>` | Delete person |
| POST | `/api/library/match` | Match against library |

---

*Last updated: February 2026*
