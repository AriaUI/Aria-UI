# Copyright 2024 The Aria-UI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from io import BytesIO
import requests
from PIL import Image
from openai import OpenAI
import numpy as np
import cv2
from android_world.agents.m3a_utils import _logical_to_physical

"""
Deploy Aria-UI with vLLM, then get the api_key and api_base from the deployment for directly API call.
"""
ariaui_api_key = "ariaui_api_key"
ariaui_api_base = "ariaui_api_base"

client = OpenAI(
    api_key=ariaui_api_key,
    base_url=ariaui_api_base,
)

models = client.models.list()
model = models.data[0].id

def encode_image_to_base64(image_path):
    pil_image = Image.open(image_path).convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str

def encode_numpy_image_to_base64(image: np.ndarray) -> str:
    """Converts a numpy array image to base64 string.
    
    Args:
        image: Numpy array representing an image (height, width, channels)
        
    Returns:
        Base64 encoded string of the image
    """
    # Convert numpy array to bytes
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image to jpg format")
    
    # Convert bytes to base64 string
    image_bytes = buffer.tobytes()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string

def request_aria_ui(image: np.ndarray, prompt: str) -> str:
    image_base64 = encode_numpy_image_to_base64(image)
    chat_completion_from_url = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        }],
        model=model,
        max_tokens=512,
        stop=["<|im_end|>"],
        extra_body= {
            "split_image": True,
            "image_max_size": 980
        }
    )

    result = chat_completion_from_url.choices[0].message.content
    print(f"Chat completion output:{result}")
    return result


def add_ui_element_mark_coords(
    screenshot: np.ndarray,
    coords: tuple[int, int],  # Normalized coordinates in [0, 1000]
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
):
    """Add a red circle marker at the specified normalized coordinates.

    Args:
        screenshot: The screenshot as a numpy ndarray.
        coords: Normalized coordinates (x, y) in range [0, 1000].
        logical_screen_size: The logical screen size.
        physical_frame_boundary: The physical coordinates in portrait orientation
          for the upper left and lower right corner for the frame.
        orientation: The current screen orientation.
    """
    # Convert normalized coordinates to logical coordinates
    logical_point = (
        coords[0] * logical_screen_size[0] // 1000,
        coords[1] * logical_screen_size[1] // 1000
    )
    
    # Convert to physical coordinates
    physical_point = _logical_to_physical(
        logical_point,
        logical_screen_size,
        physical_frame_boundary,
        orientation,
    )

    # Draw a large red circle
    radius = 30  # Adjust size as needed
    cv2.circle(
        screenshot,
        physical_point,
        radius,
        color=(0, 0, 255),  # BGR format - Red
        thickness=3
    )

def convert_coords_to_physical(coords: tuple[int, int], logical_screen_size: tuple[int, int], physical_frame_boundary: tuple[int, int, int, int], orientation: int) -> tuple[int, int]:
    logical_point = (
        coords[0] * logical_screen_size[0] // 1000,
        coords[1] * logical_screen_size[1] // 1000
    )
    physical_point = _logical_to_physical(
        logical_point,
        logical_screen_size,
        physical_frame_boundary,
        orientation,
    )
    return physical_point