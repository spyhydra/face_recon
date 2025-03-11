# Docker Deployment Guide for Face Recognition Attendance System

## Project Structure Analysis

The Face Recognition Attendance System consists of several key components:

### Core Components:
1. **Face Recognition Engine**
   - `advanced_face_recognition.py`: Advanced face detection and recognition
   - `face_utils.py`: Traditional face detection utilities
   - Uses both dlib and OpenCV for robust face detection

2. **Data Management**
   - `db_utils.py`: Database operations for student records
   - Uses CSV files for storing student details and attendance

3. **User Interface**
   - `main_app.py`: Main GUI application using tkinter
   - `ui_utils.py`: UI utility functions

4. **Image Processing**
   - `takeImage.py`: Capture and process student images
   - `trainImage.py`: Train traditional recognition model
   - `train_advanced_model.py`: Train advanced recognition model

### Directory Structure:
```
/app
├── TrainingImage/          # Student face images
├── StudentDetails/         # Student information
├── Attendance/            # Attendance records
├── TrainingImageLabel/    # Trained models
└── UI_Image/             # UI assets
```

## Prerequisites

1. **Docker Installation**
   - Install Docker Engine
   - Install Docker Compose
   - Ensure X11 forwarding is set up (for GUI)

2. **System Requirements**
   - Linux/Unix system with X11 server
   - Webcam access
   - At least 4GB RAM
   - 10GB free disk space

## Installation Steps

1. **Prepare the Environment**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd attendance-system
   
   # Create necessary directories
   mkdir -p TrainingImage StudentDetails Attendance TrainingImageLabel
   ```

2. **Configure X11 Forwarding**
   ```bash
   # On Linux host
   xhost +local:docker
   
   # Set DISPLAY variable
   export DISPLAY=:0
   ```

3. **Build and Run**
   ```bash
   # Build the Docker image
   docker-compose build

   # Run the application
   docker-compose up
   ```

## Configuration

### 1. Environment Variables
The following environment variables can be modified in `docker-compose.yml`:
- `DISPLAY`: X11 display server
- `PYTHONUNBUFFERED`: Python output buffering
- `QT_X11_NO_MITSHM`: X11 shared memory settings

### 2. Volume Mounts
The following directories are mounted as volumes:
- `./TrainingImage:/app/TrainingImage`
- `./StudentDetails:/app/StudentDetails`
- `./Attendance:/app/Attendance`
- `./TrainingImageLabel:/app/TrainingImageLabel`

### 3. Device Access
The webcam device is mapped into the container:
```yaml
devices:
  - /dev/video0:/dev/video0
```

## Troubleshooting

### 1. GUI Issues
If the GUI doesn't appear:
```bash
# Check X11 permissions
xhost +local:docker

# Verify DISPLAY variable
echo $DISPLAY
```

### 2. Webcam Access
If the webcam isn't working:
```bash
# Check webcam device
ls -l /dev/video*

# Verify permissions
sudo usermod -aG video $USER
```

### 3. Performance Issues
If experiencing slow performance:
```bash
# Check container resources
docker stats face-recognition-app

# Adjust batch size in gpu_accelerated.py
BATCH_SIZE = 4  # Reduce if needed
```

## Maintenance

### 1. Backup Data
Regularly backup mounted volumes:
```bash
# Backup script
tar -czf backup.tar.gz TrainingImage StudentDetails Attendance TrainingImageLabel
```

### 2. Log Management
Container logs are available via:
```bash
docker-compose logs face-recognition-app
```

### 3. Updates
To update the application:
```bash
# Pull latest changes
git pull

# Rebuild container
docker-compose build --no-cache
docker-compose up -d
```

## Security Considerations

1. **Data Protection**
   - All student data is stored in mounted volumes
   - Implement regular backups
   - Set appropriate file permissions

2. **Container Security**
   - Container runs with limited privileges
   - Only necessary ports and devices are exposed
   - Regular security updates for base image

3. **Access Control**
   - X11 forwarding is restricted to local connections
   - Webcam access is controlled via device mapping

## Resource Requirements

1. **Container Resources**
   - CPU: 2+ cores recommended
   - RAM: 4GB minimum
   - Storage: 10GB minimum
   - GPU: Optional, but recommended for advanced recognition

2. **Host System**
   - X11 server
   - Webcam support
   - Docker Engine 19.03+
   - Docker Compose 1.27+

## Development Notes

1. **Building for Development**
   ```bash
   # Build with development tools
   docker-compose -f docker-compose.dev.yml build
   ```

2. **Testing**
   ```bash
   # Run tests
   docker-compose exec face-recognition-app python -m pytest
   ```

3. **Debugging**
   ```bash
   # Access container shell
   docker-compose exec face-recognition-app bash
   
   # Check logs
   docker-compose logs -f face-recognition-app
   ``` 