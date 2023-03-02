
from tflite_runtime.interpreter import Interpreter
from PIL import Image

import numpy as np
import argparse
import picamera

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def load_image(frame, new_size=(224, 224)):
  #Function to crop, convery to numpy and normalize the input frame into a image
  
  # Get the dimensions
  height, width, _ = frame.shape # Image shape
  new_width, new_height = new_size # Target shape 

  # Calculate the target image coordinates
  left = (width - new_width) // 2
  top = (height - new_height) // 2
  right = (width + new_width) // 2
  bottom = (height + new_height) // 2

  image = frame[left: right, top: bottom, :]
  
  # Convert to Numpy with float32 as the datatype
  image = np.array(image, dtype=np.float32)
  
  # Normalize the image
  image = image / 255.0

  return image

parser = argparse.ArgumentParser(description='Rock, paper and scissors')
parser.add_argument('--model_path', type=str, help='Specify the model path', required=True)

args = parser.parse_args()
model_path = args.model_path 

labels = ['Rock', 'Paper', 'Scissors']

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path= model_path)

#Allocate tensors to the interpreter
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input size
input_shape = input_details[0]['shape']
input_size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

# Prepare for display on screen
plt.ion()
plt.tight_layout()
    
fig = plt.gcf()
fig.canvas.set_window_title('Object Detection')
fig.suptitle('Detecting')
ax = plt.gca()
ax.set_axis_off()
tmp = np.zeros(input_size + [3], np.uint8)
preview = ax.imshow(tmp)
    

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    while True:
        # Capture a frame from camera
        stream = np.empty((480, 640, 3), dtype=np.uint8)
        camera.capture(stream, 'rgb')

        # Function to crop, convery to numpy and normalize the input frame into a image
        image = load_image(stream)

        # Add a batch dimension
        input_data = np.expand_dims(image, axis= 0)

        # Point the data to be used for testing and run the interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Obtain results and print the predicted category
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the label with highest probability
        predicted_label = np.argmax(predictions)
        
        # Print the predicted category
        print(labels[predicted_label])

        preview.set_data(image)
        fig.canvas.get_tk_widget().update()
            
