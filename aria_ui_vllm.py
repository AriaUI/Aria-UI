from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ast
from utils import draw_coord


model_path = "Aria-UI/Aria-UI-base"

def main():
    llm = LLM(
        model=model_path,
        tokenizer_mode="slow",
        dtype="bfloat16",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    instruction = "Try Aria."
    image_path = "examples/aria.png"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: " + instruction,
                }
            ],
        }
    ]

    message = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    outputs = llm.generate(
        {
            "prompt_token_ids": message,
            "multi_modal_data": {
                "image": [
                    Image.open(image_path),
                ],
                "max_image_size": 980,  # [Optional] The max image patch size, default `980`, maximum `980`, the image size for splitted blocks
                "split_image": True,  # [Optional] whether to split the images, default `True`
            },
        },
        sampling_params=SamplingParams(max_tokens=50, top_k=1, stop=["<|im_end|>"]),
    )

    for o in outputs:
        generated_tokens = o.outputs[0].token_ids
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(response)
        coords = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
        image = draw_coord(Image.open("examples/aria.png"), coords)
        image.save("output.png")


if __name__ == "__main__":
    main()
