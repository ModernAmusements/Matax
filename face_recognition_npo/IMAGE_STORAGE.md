# NGO Facial Image Analysis System - Image Storage Guide

## Storage Structure

### Reference Images
**Location**: `reference_images/`
**Purpose**: Store reference embeddings for comparison

```
reference_images/
├── embeddings.json    # Stored embeddings (NOT raw images!)
└── README.md
```

**Format**: `embeddings.json`
```json
{
  "metadata": [
    {"id": "name", "path": "path/to/image.jpg", "added_at": "timestamp"}
  ],
  "embeddings": [
    {"id": "name", "embedding": [0.1, 0.5, ...]}  // 512-dim vector
  ]
}
```

### Test Images
**Location**: `test_images/`
**Purpose**: Test images for development

```
test_images/
├── test_subject.jpg
├── reference_subject.jpg
└── README.md
```

---

## What Gets Stored

### Embeddings (Not Raw Images!)
- 512-dimensional vectors (ArcFace) or 128-dim (FaceNet)
- Non-reversible mathematical representation
- Cannot reconstruct original face from embedding

### Metadata
- Reference ID (unique identifier)
- Path to original image
- Timestamp

---

## Privacy Protection

✅ **No raw biometric data stored**
✅ **Only mathematical representations**
✅ **Original images referenced by path only**

---

## Best Practices

1. **Store embeddings only** - No raw face images in JSON
2. **Delete embeddings** when no longer needed
3. **Track consent** in separate system
4. **Regular backups** of `embeddings.json`

---

## File Management

```bash
# List references
curl http://localhost:3000/api/references

# Clear all references
curl -X POST http://localhost:3000/api/clear
```

The system stores only embeddings, not raw images!
