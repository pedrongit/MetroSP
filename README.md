# Final Project: Turnstile Control with Computer Vision and OpenVINO

This project uses Python with OpenVINO for person detection and Arduino for controlling a turnstile. Communication between Python and Arduino is done via serial port, where the Arduino receives commands to open or close the turnstile based on person detection.

## Components Used

- **Arduino**: Controls the relay to open/close the turnstile.
- **Python**: Utilizes OpenVINO for video processing and person detection.
- **OpenVINO**: Inference engine for computer vision models.
- **Relay**: Controls the turnstile power.
- **Camera**: Captures video for processing.

## Arduino Code

The Arduino code configures the device to control a relay based on commands received via the serial port.

## Python Code

The Python code uses the OpenVINO library to perform person detection in a video. Based on the detection, it sends commands to the Arduino via the serial port to open or close the turnstile.

## Project Setup

### Prerequisites

- Arduino IDE
- Python 3.x
- OpenVINO Toolkit
- OpenCV
- pySerial

### Installation

1. **Arduino**:
    - Upload the provided Arduino code to your Arduino board.
    - Ensure the relay is connected to the defined pin.

2. **Python**:
    - Install the required Python libraries:
      ```bash
      pip install opencv-python-headless openvino pyserial numpy
      ```
    - Download and install the OpenVINO toolkit from [OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).

### Running the Project

1. Connect the Arduino to your computer.
2. Run the Python script to start video processing and turnstile control:
   ```bash
   python main.py
3.The script will open a video window where the person detection and turnstile status will be displayed.

Notes
Adjust the video source path in the Python script if necessary.
Ensure the serial port defined in the Python script matches the one used by the Arduino.
