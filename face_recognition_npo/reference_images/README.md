# Reference Images Folder

This folder stores reference images added through the Electron UI.

## Files

| File/Folder | Purpose |
|-------------|---------|
| `embeddings.json` | Stores reference metadata and 128-dim embeddings (auto-managed) |
| `README.md` | This file |

## What Gets Stored

When you add a reference through the UI, the following is stored in `embeddings.json`:

```json
{
  "metadata": [
    {
      "id": 0,
      "name": "subject.jpg",
      "thumbnail": "base64_encoded_thumbnail...",
      "added_at": "2026-02-11 14:30:00.123456"
    }
  ],
  "embeddings": [
    {
      "id": 0,
      "embedding": [0.123, -0.456, ...]  // 128-dimensional vector
    }
  ]
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique identifier (auto-assigned, 0-indexed) |
| `name` | string | Original filename |
| `thumbnail` | string | Base64-encoded thumbnail (160x160) |
| `embedding` | array | 128-dimensional vector (NOT the image) |
| `added_at` | string | Timestamp when added |

## Privacy

- ✅ Only embeddings are stored (128 floats ≈ 512 bytes)
- ✅ Original images are NOT copied - only referenced by path
- ✅ Embeddings are non-reversible (cannot reconstruct face)
- ✅ Consent status recorded
- ✅ Timestamps for audit trail

## Usage

References are managed through the Electron UI:

1. Click "+ Add Reference"
2. Select an image
3. Face is detected and embedding extracted
4. Reference saved to `embeddings.json`
5. Persists across app restarts

## Cleanup

To reset references:

```bash
# Option 1: Delete embeddings.json (creates empty one on next save)
rm reference_images/embeddings.json

# Option 2: Clear via API
curl -X POST http://localhost:3000/api/clear
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/references` | List all references |
| POST | `/api/add-reference` | Add new reference |
| DELETE | `/api/references/<id>` | Remove reference by ID |
| GET | `/api/visualizations/<type>/reference/<id>` | Get visualization for ref |
