# Installation Guide for Face Recognition Attendance System

This guide will help you set up the Face Recognition Attendance System with GPU acceleration support.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM
- Webcam
- Windows 10 or higher

### For GPU Acceleration
- NVIDIA GPU with CUDA support
- 4GB GPU RAM (GTX 1650 or better)
- CUDA Toolkit 11.8 or higher
- Visual Studio Build Tools 2019 or higher

## Step-by-Step Installation

### 1. Install Python
1. Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. During installation, make sure to check "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version
   ```

### 2. Install CUDA Toolkit (for GPU support)
1. Download CUDA Toolkit 11.8 from [NVIDIA's website](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Follow the installation wizard
3. Verify installation:
   ```bash
   nvcc --version
   ```

### 3. Install Visual Studio Build Tools
1. Download Visual Studio Build Tools 2019 or higher
2. Select "Desktop development with C++"
3. Install the selected components

### 4. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 5. Install Dependencies

#### Basic Installation (CPU-only)
```bash
pip install -r requirements.txt
```

#### GPU-enabled Installation
```bash
# First, install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

### 6. Install dlib with CUDA Support
```bash
# Install CMake (required for dlib)
pip install cmake

# Install dlib
pip install dlib
```

### 7. Verify Installation

Run the GPU test script to verify everything is working:
```bash
python test_gpu_acceleration.py
```

## Common Issues and Solutions

### 1. CUDA Not Found
If you see "CUDA not available" when running the program:
- Verify CUDA Toolkit installation
- Check if your GPU is CUDA-compatible
- Update NVIDIA drivers

### 2. dlib Installation Fails
If dlib installation fails:
- Make sure Visual Studio Build Tools are installed
- Try installing the pre-built wheel:
  ```bash
  pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
  ```

### 3. Import Error: DLL load failed
If you see DLL load failed errors:
- Install Visual C++ Redistributable
- Ensure all PATH environment variables are set correctly

### 4. Low Performance
If face recognition is slow:
- Check GPU utilization in Task Manager
- Adjust batch size in `gpu_accelerated.py`
- Close other GPU-intensive applications

## Configuration

### Adjusting GPU Settings
You can modify GPU-related settings in `config.py`:
```python
# Example GPU configuration
GPU_BATCH_SIZE = 8  # Adjust based on your GPU memory
GPU_ENABLED = True  # Enable/disable GPU acceleration
```

### Memory Management
If you encounter CUDA out of memory errors:
1. Reduce `BATCH_SIZE` in `gpu_accelerated.py`
2. Lower image resolution in `config.py`
3. Close other GPU-intensive applications

## Testing the Installation

After installation, run these tests to verify everything is working:

1. Test GPU Detection:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

2. Test Face Recognition:
```bash
python test_gpu_acceleration.py
```

3. Test Full System:
```bash
python main_app.py
```

## Updating

To update the system to the latest version:
```bash
git pull
pip install -r requirements.txt --upgrade
```

## Support

If you encounter any issues:
1. Check the troubleshooting section in this guide
2. Look for similar issues in the project's issue tracker
3. Create a new issue with detailed information about your problem

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [dlib Installation Guide](http://dlib.net/compile.html)
- [Face Recognition Library Documentation](https://face-recognition.readthedocs.io/) 