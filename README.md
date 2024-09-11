# Monolayer-Detection
## Dependencies
### Camera: 
tisgrabber.py, tisgrabber_x64.dll, and TIS_UDSHL11_x64.dll in the versions used here: https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python/tisgrabber/samples
### Motor (BDD 202): 
From Thorlabs Kinesis package, which should be installed in C:/Program Files/Thorlabs/Kinesis:
- Thorlabs.MotionControl.DeviceManageCLI.dll
- Thorlabs.MotionControl.GenericMotorCLI.dll
- Thorlabs.MotionControlBenchtop.BrushlessMotorCLI.dll

### Python libraries that need to be installed:
- ultralytics and all dependencies(Pytorch version 2.4.0 does not work on Windows, a lower version like 2.3.1 works)
  - libomp140x86_64.dll was missing in pytorch installation - usually fixed by installing C++ build tools on an updated version of Visual Studio
- pythonnet
### Model Weights:
best.pt: weights from a pretrained YOLO model

## Usage:
edges.py is the primary program
- Open command prompt
- Type these commands:
  - cd Documents/Karolyn
  - activate Karolyn (activate conda environment with necessary installations)
  - python edges.py (run the program)
- A live video feed will pop up on the screen
- The program first asks the user to take a picture of a blank background: this only needs to be done once per setup (can skip by pressing Enter)
- The program then asks the user to place the bottom right corner of the chip in view of the camera
  - This assumes the camera is set up so that moving the motor to the right wll move the view of the camera to the right as well (looks like chip moves to the left)
  - The motor moves to the right as its first move, so the edge should still be visible after this move
- The program scans the chip, and saves the motor position of the bottom right corner, as well as the positions of all found monolayers, in output.txt
- output.txt is overwritten every time the program is run

