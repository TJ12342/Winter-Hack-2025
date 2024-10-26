
Project repositories:TJ12342/Winter-Hack-2025

Program Introduction:

This program is designed to detect laser pointers of various colors (red, green, and blue) using a webcam feed. By applying color thresholds in the HSV color space, it identifies and marks the position of the laser pointer on the screen.

Usage Guide

Installation:

Ensure you have Python installed on your system.

Install the necessary libraries using pip:

pip install opencv-python numpy

Running the Program:

Run the program using the command:

python main.py

Operation:

Once the program runs, your webcam will activate, and a window will display the webcam feed.

Move a red, green, or blue laser pointer in front of the webcam.

The program marks the detected laser point with a green circle on the screen.

Exiting the Program:

Press the ‘q’ key on your keyboard to quit the program and close the webcam feed window.

Note

Ensure good lighting conditions to improve detection accuracy.

If needed, adjust the HSV color range in the code to better match your specific laser pointer colors, as variations in lighting and camera characteristics might affect detection.

