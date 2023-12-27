### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_google_net_model
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ['Normal', 'Pneumonia', 'Tuberculosis']

google_net_model, google_net_transforms = create_google_net_model(num_classes = len(class_names))
# Load saved weights
google_net_model.load_state_dict(
                              torch.load(f='x-ray.pth',
                              map_location=torch.device('cpu')))


# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = google_net_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    google_net_model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(google_net_model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


# Create title, description and article strings
title = "Pneumonia|Tuberculosis detector"
description = "A Google_net based feature extractor computer vision model to classify images of an x-ray."
article = "Created at google colab"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
