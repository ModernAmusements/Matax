# Sample Reference Images for NGO Facial Image Analysis System

This directory contains sample reference images for testing and demonstration purposes.

## Images

### test_reference_1.jpg
- **Description**: Male face, frontal view, good lighting
- **Use Case**: Basic face detection test
- **Metadata**: 
  - Name: John Doe
  - Document ID: DOC001
  - Consent: Yes
  - Source: Test document

### test_reference_2.jpg  
- **Description**: Female face, slight angle, moderate lighting
- **Use Case**: Pose variation test
- **Metadata**: 
  - Name: Jane Smith
  - Document ID: DOC002
  - Consent: Yes
  - Source: Test photo

### test_reference_3.jpg
- **Description**: Male face, profile view, challenging lighting
- **Use Case**: Edge case detection test
- **Metadata**: 
  - Name: Michael Brown
  - Document ID: DOC003
  - Consent: Yes
  - Source: Test scan

### test_reference_4.jpg
- **Description**: Female face, multiple people, partial occlusion
- **Use Case**: Multiple face detection test
- **Metadata**: 
  - Name: Sarah Johnson
  - Document ID: DOC004
  - Consent: Yes
  - Source: Group photo

### test_reference_5.jpg
- **Description**: Male face, low resolution, poor lighting
- **Use Case**: Quality variation test
- **Metadata**: 
  - Name: Robert Wilson
  - Document ID: DOC005
  - Consent: Yes
  - Source: Low quality document

## Usage

### Add Reference Images
```bash
python3 main.py reference add reference_images/test_reference_1.jpg john_doe --metadata '{"name": "John Doe", "document_id": "DOC001", "consent": "yes", "source": "test_document"}'
python3 main.py reference add reference_images/test_reference_2.jpg jane_smith --metadata '{"name": "Jane Smith", "document_id": "DOC002", "consent": "yes", "source": "test_photo"}'
```

### Test Detection
```bash
python3 main.py detect reference_images/test_reference_1.jpg
python3 main.py detect reference_images/test_reference_4.jpg  # Should detect multiple faces
```

### Test Verification
```bash
python3 main.py verify reference_images/test_reference_1.jpg  # Should match reference 1
python3 main.py verify reference_images/test_reference_3.jpg  # Should match reference 3
```

## Metadata Structure

Each reference image should have the following metadata:

```json
{
  "name": "Full Name",
  "document_id": "Unique ID",
  "consent": "yes|no|unknown",
  "source": "document|photo|scan|other",
  "added_at": "YYYY-MM-DD",
  "notes": "Additional information"
}
```

## Image Quality Guidelines

### Good Quality Images
- Clear frontal view
- Good lighting
- High resolution
- No occlusion

### Challenging Cases
- Profile views
- Poor lighting
- Low resolution
- Partial occlusion
- Multiple faces

## Test Scenarios

### Basic Tests
1. **Single face detection** - test_reference_1.jpg
2. **Multiple face detection** - test_reference_4.jpg
3. **Pose variation** - test_reference_2.jpg, test_reference_3.jpg

### Edge Cases
1. **Low quality** - test_reference_5.jpg
2. **Challenging lighting** - test_reference_3.jpg
3. **Partial occlusion** - test_reference_4.jpg

### Performance Tests
1. **Batch processing** - multiple images
2. **Real-time processing** - webcam demo
3. **Large reference set** - all 5 references

## Notes

- These images are for testing purposes only
- In production, use actual reference images with proper consent
- Always maintain documentation of consent and source
- Regularly update reference images for better accuracy

The sample images demonstrate various real-world scenarios that NGOs might encounter in documentation verification workflows.