import torch
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import ast
from utils import draw_coord, resize_image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "Aria-UI/Aria-UI-base"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_file = "./examples/aria.png"
instruction = "Try Aria."

image = Image.open(image_file).convert("RGB")

# NOTE: using huggingface on a single 80GB GPU, we resize the image to 1920px on the long side to prevent OOM. this is unnecessary with vllm.
image = resize_image(image, long_size=1920)

messages = [
    {
        "role": "user",
        "content": [
            {"text": None, "type": "image"},
            {"text": instruction, "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        # do_sample=True,
        # temperature=0.9,
    )

output_ids = output[0][inputs["input_ids"].shape[1] :]
response = processor.decode(output_ids, skip_special_tokens=True)
print(response)

coords = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
# Save the image with the predicted coordinates
image = draw_coord(image, coords)
image.save("output.png")