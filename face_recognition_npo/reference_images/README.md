# Reference Images Folder

This folder stores reference images added through the Electron UI.

## Files

| File/Folder | Purpose |
|-------------|---------|
| `embeddings.json` | Stores reference metadata and embeddings (auto-managed) |
| `*/` | Subdirectories (if any - usually empty) |

## What Gets Stored

When you add a reference through the UI, the following is stored in `embeddings.json`:

```json
{
  "metadata": [
    {
      "id": "auto-generated-id",
      "path": "path/to/image.jpg",
      "metadata": {
        "source": "upload|example|test",
        "consent": true
      },
      "added_at": "2026-02-10 10:55:11.029069"
    }
  ],
  "embeddings": []
}
```

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Auto | Unique identifier (auto-generated) |
| `path` | Auto | Where the image was loaded from |
| `source` | Optional | Where image came from (upload/example/test) |
| `consent` | Yes | Whether consent was obtained (true/false) |
| `added_at` | Auto | Timestamp when added |

**Note:** No name or document_id is stored. Use the UI to manage references.

## Usage

References are managed entirely through the Electron UI:

1. Click "+ Add Reference"
2. Select an image
3. Consent is recorded
4. Reference is saved to `embeddings.json`

## Cleanup

The `kanye_west/` subfolder is leftover from old tests and can be safely deleted:

```bash
rm -rf reference_images/kanye_west/
```

## Privacy

- No personal names or document IDs are stored
- Only 128-dimensional embeddings (non-reversible)
- Consent status is recorded
- Timestamps track when references were added
