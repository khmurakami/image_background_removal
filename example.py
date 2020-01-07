#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Default Libraries
import os

# Company Libraries
from background_helper import BackgroundRemoval

# Load the model
model_path = "/home/kmurakami/tshirt/tshirt-airflow/lib/background_removal/mobile_net_model/frozen_inference_graph.pb"

# Instantiate the model
model_client = BackgroundRemoval(model_path)

# Run the image
model_client.run_visualization("tiffany_test.jpg", "output.png")