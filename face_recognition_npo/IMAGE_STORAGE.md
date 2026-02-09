# NGO Facial Image Analysis System - Image Storage Guide

## **Image Storage Structure**

### **1. Reference Images Directory**
**Location**: `reference_images/`
**Purpose**: Store reference images for comparison (with consent)

```
reference_images/
├── john_doe.jpg          # Reference image 1
├── jane_smith.jpg       # Reference image 2  
├── michael_brown.jpg    # Reference image 3
└── README.md            # Documentation
```

### **2. Test Images Directory**
**Location**: `test_images/`
**Purpose**: Store test images for development and testing

```
test_images/
├── test_image_1.jpg     # Test image 1
├── test_image_2.jpg     # Test image 2
├── verification_1.jpg   # Verification image 1
└── README.md            # Documentation
```

### **3. Captured Faces Directory**
**Location**: `captured_faces/`
**Purpose**: Store faces captured from webcam for testing

```
captured_faces/
├── captured_face_1.jpg  # Face captured 1
├── captured_face_2.jpg  # Face captured 2
└── README.md            # Documentation
```

## **How to Use These Directories**

### **Add Reference Images**
```bash
# Add images to reference_images directory
cp john_doe.jpg reference_images/
cp jane_smith.jpg reference_images/

# Then add them to the system
python3 main.py reference add reference_images/john_doe.jpg john_doe --metadata '{"name": "John Doe", "consent": "yes"}'
python3 main.py reference add reference_images/jane_smith.jpg jane_smith --metadata '{"name": "Jane Smith", "consent": "yes"}'
```

### **Test with Images**
```bash
# Copy test images
cp test_image.jpg test_images/

# Test detection
python3 main.py detect test_images/test_image.jpg

# Test verification
python3 main.py verify test_images/test_image.jpg
```

### **Capture from Webcam**
```bash
# Captured faces will be saved automatically
python3 main.py webcam
# Press 'c' to capture - images saved to captured_faces/
```

## **Image File Naming Convention**

### **Reference Images**
```
<unique_id>.jpg          # e.g., john_doe.jpg, jane_smith.jpg
```

### **Test Images**
```
<test_name>_<number>.jpg    # e.g., test_image_1.jpg, verification_2.jpg
```

### **Captured Faces**
```
captured_<timestamp>.jpg  # e.g., captured_20231201_153045.jpg
```

## **Metadata Structure**

Each reference image should have metadata:

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

## **Best Practices**

### **1. Organize by Category**
```
reference_images/
├── documents/           # Scanned documents
├── photos/              # Personal photos  
├── ids/                 # ID cards/passports
└── other/               # Other sources
```

### **2. Version Control**
```
reference_images/
├── john_doe_v1.jpg      # Original version
├── john_doe_v2.jpg      # Updated version
└── john_doe_current.jpg # Current active version
```

### **3. Backup Strategy**
```
reference_images/
├── active/              # Current working images
├── archive/             # Old versions
└── backups/             # Regular backups
```

## **File Management Commands**

### **Copy Images**
```bash
# Copy single image
cp /path/to/image.jpg reference_images/

# Copy multiple images
cp /path/to/images/*.jpg reference_images/
```

### **Move Images**
```bash
# Move image to reference directory
mv /path/to/image.jpg reference_images/
```

### **List Images**
```bash
# List all reference images
ls -la reference_images/

# List with details
ls -la reference_images/*.jpg
```

## **Storage Recommendations**

### **For Small NGOs**
- Keep all images in `reference_images/`
- Use simple naming convention
- Regular manual backups

### **For Medium NGOs**
- Organize by source/department
- Use metadata for tracking
- Automated backup system

### **For Large NGOs**
- Database-driven image management
- Cloud storage integration
- Advanced access controls

## **Security Considerations**

### **1. Access Control**
- Restrict access to reference_images directory
- Use file permissions appropriately
- Consider encryption for sensitive images

### **2. Backup Strategy**
- Regular automated backups
- Off-site storage for critical images
- Version control for important images

### **3. Privacy Protection**
- Store only necessary images
- Delete images when no longer needed
- Follow data protection regulations

## **Troubleshooting**

### **Image Not Found**
```bash
# Check if image exists
ls reference_images/

# Check file permissions
ls -la reference_images/
```

### **Permission Denied**
```bash
# Change directory permissions
chmod 755 reference_images/

# Change file permissions
chmod 644 reference_images/*.jpg
```

### **Disk Space Issues**
```bash
# Check disk usage
df -h

# Clean up old images
find reference_images/ -name "*.jpg" -mtime +365 -delete
```

The directories are now created and ready for use! You can start adding your reference images to `reference_images/` and test images to `test_images/`.