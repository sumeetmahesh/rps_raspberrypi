# Live Rock Paper Scissor Detection in Raspberry Pi

Rock, paper, scissor classifier to classify live camera feed on Raspberry Pi.

Four files:

1. RPS_Camera.ipyb - Used to create the rock, paper, scissor classification model in Jupyter notebook on work station or Colabs. I used transfer learning for the model using mobilenet_v2 as feature extractor and adding one dense layer for classification. This code also saves the retrained model and coverts to tflite format.
2. rps.tflite - Rock Paper Scissor tflite model. 
3. rps_main.py - Python script to execute rps.tflite on Raspberry Pi from command line.
4. README.md - Instructions for command line execution.

## Prerequisites
* PiCamera  
* matplotlib (with tkinter as backend)
* python-tk  
* Pillow  
* numpy

To install the Python dependencies, run:
```
pip install -r requirements.txt
```

Next, to run the code on Raspberry Pi, use `classify.py` as follows:

```
python3 rps_main.py --model_path rps.tflite 
```


