# Live Rock Paper Scissor Detection in Raspberry Pi

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


