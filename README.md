# Voice Controlled Robot Arm

Here contains the code for a voice controlled robot arm. The hardware consists of a raspberry pi, arduino and an assembled mechanical arm 

## Folders
- ### Commands
Here lies the code necessary for loading the commands audio data and converting it to numpy arrays, the model definition and training of the commands model.
- ### Preprocessing
Here lies the code necessary for preprocessing of audio data including making sure all audio clips are of the same length, making
the negative data and summing up all the numpy array batches made by triggerword.

- ### Triggerword
Here lies the code necessary for loading the triggerwords audio data and converting it to numpy arrays, the triggerword model definition and training of the triggerword model and utilities for convertion of the wav files into spectograms

- ### Practice_code
Here lies all the miscellaneous bits of code I came up with to complete the project, scripts and code from here is aggregrated to form the scripts that actually run the project

## Usage
Connect the arduino to the raspberry pi, with the mechanical arm connected in the necessary arduino ports. The run ``` python3 rasptestrealtime.py ```
