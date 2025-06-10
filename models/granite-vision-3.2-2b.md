# granite-vision-3.2-2b

**Model Summary:**
granite-vision-3.2-2b is a compact and efficient vision-language model, specifically designed for visual document understanding, enabling automated content extraction from tables, charts, infographics, plots, diagrams, and more. The model was trained on a meticulously curated instruction-following dataset, comprising diverse public datasets and synthetic datasets tailored to support a wide range of document understanding and general image tasks. It was trained by fine-tuning a Granite large language model with both image and text modalities.


**Evaluations:**

We evaluated Granite Vision 3.2 alongside other vision-language models (VLMs) in the 1B-4B parameter range using the standard llms-eval benchmark. The evaluation spanned multiple public benchmarks, with particular emphasis on document understanding tasks while also including general visual question-answering benchmarks. 

|  | Molmo-E | InternVL2 | Phi3v | Phi3.5v | Granite Vision |
|-----------|--------------|----------------|-------------|------------|------------|
| **Document benchmarks** |
| DocVQA | 0.66 | 0.87 | 0.87 | 0.88 | **0.89** |
| ChartQA | 0.60 | 0.75 | 0.81 | 0.82 | **0.87** |
| TextVQA | 0.62 | 0.72 | 0.69 | 0.7 | **0.78** |
| AI2D | 0.63 | 0.74 | **0.79** | **0.79**  | 0.76 |
| InfoVQA | 0.44 | 0.58 | 0.55 | 0.61  | **0.64** |
| OCRBench | 0.65 | 0.75 | 0.64 | 0.64 | **0.77** |
| LiveXiv VQA | 0.47 | 0.51 | **0.61** | - | **0.61** |
| LiveXiv TQA | 0.36 | 0.38 | 0.48 | - | **0.57** |
| **Other benchmarks** |
| MMMU | 0.32 | 0.35 | 0.42 | **0.44** | 0.37 |
| VQAv2 | 0.57 | 0.75 | 0.76 | 0.77 | **0.78** |
| RealWorldQA | 0.55 | 0.34 | 0.60 | 0.58 | **0.63** |
| VizWiz VQA | 0.49 | 0.46 | 0.57 | 0.57 |  **0.63** |
| OK VQA | 0.40 | 0.44 | 0.51 | 0.53 | **0.56** |


- **Paper:** [Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence](https://arxiv.org/abs/2502.09927)
- **Release Date**: Feb 26th, 2025 
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Input Format:**
Currently the model supports English instructions and images (png, jpeg, etc.) as input format. 

**Intended Use:** 
The model is intended to be used in enterprise applications that involve processing visual and text data. In particular, the model is well-suited for a range of visual document understanding tasks, such as analyzing tables and charts, performing optical character recognition (OCR), and answering questions based on document content. Additionally, its capabilities extend to general image understanding, enabling it to be applied to a broader range of business applications. For tasks that exclusively involve text-based input, we suggest using our Granite large language models, which are optimized for text-only processing and offer superior performance compared to this model.


## Generation:

Granite Vision model is supported natively `transformers>=4.49`. Below is a simple example of how to use the `granite-vision-3.2-2b` model.

### Usage with `transformers`

First, make sure to build the latest version of transformers:
```shell
pip install transformers>=4.49
```

Then run the code:
```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

# prepare image and text prompt, using the appropriate prompt template

img_path = hf_hub_download(repo_id=model_path, filename='example.png')

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "What is the highest scoring model on ChartQA and what is its score?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)


# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Usage with vLLM

The model can also be loaded with `vLLM`. First make sure to install the following libraries:

```shell
pip install torch torchvision torchaudio
pip install vllm==0.6.6
```
Then, copy the snippet from the section that is relevant for your use case.

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from huggingface_hub import hf_hub_download
from PIL import Image

model_path = "ibm-granite/granite-vision-3.2-2b"

model = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=64,
)

# Define the question we want to answer and format the prompt
image_token = "<image>"
system_prompt = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

question = "What is the highest scoring model on ChartQA and what is its score?"
prompt = f"{system_prompt}<|user|>\n{image_token}\n{question}\n<|assistant|>\n"
img_path = hf_hub_download(repo_id=model_path, filename='example.png')
image = Image.open(img_path).convert("RGB")
print(image)

# Build the inputs to vLLM; the image is passed as `multi_modal_data`.
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image,
    }
}

outputs = model.generate(inputs, sampling_params=sampling_params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

### Fine-tuning

For an example of fine-tuning Granite Vision for new tasks refer to [this notebook](https://huggingface.co/learn/cookbook/en/fine_tuning_granite_vision_sft_trl).

### Use Granite Vision for MM RAG

For an example of MM RAG using granite vision refer to [this notebook](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/RAG/Granite_Multimodal_RAG.ipynb).

**Model Architecture:** 

The architecture of granite-vision-3.2-2b consists of the following components:

(1) Vision encoder: SigLIP (https://huggingface.co/docs/transformers/en/model_doc/siglip).

(2) Vision-language connector: two-layer MLP with gelu activation function.

(3) Large language model: granite-3.1-2b-instruct with 128k context length (https://huggingface.co/ibm-granite/granite-3.1-2b-instruct).

We built upon LLaVA (https://llava-vl.github.io) to train our model. We use multi-layer encoder features and a denser grid resolution in AnyRes to enhance the model's ability to understand nuanced visual content, which is essential for accurately interpreting document images. 


**Training Data:** 

Overall, our training data is largely comprised of two key sources: (1) publicly available datasets (2) internally created synthetic data targeting specific capabilities including document understanding tasks. A detailed attribution of datasets can be found on our [github page](https://github.com/ibm-granite/granite-vision-models) and also in the [technical report](https://arxiv.org/abs/2502.09927).


**Infrastructure:**
We train Granite Vision using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Ethical Considerations and Limitations:** 
The use of Large Vision and Language Models involves risks and ethical considerations people must be aware of, including but not limited to: bias and fairness, misinformation, and autonomous decision-making. granite-vision-3.2-2b is not the exception in this regard. Although our alignment processes include safety considerations, the model may in some cases produce inaccurate, biased, or unsafe responses to user prompts.
Additionally, it remains uncertain whether smaller models might exhibit increased susceptibility to hallucination in generation scenarios due to their reduced sizes, which could limit their ability to generate coherent and contextually accurate responses.
This aspect is currently an active area of research, and we anticipate more rigorous exploration, comprehension, and mitigations in this domain. Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use granite-vision-3.2-2b with ethical intentions and in a responsible way. We recommend using this model for document understanding tasks, and note that more general vision tasks may pose higher inherent risks of triggering biased or harmful output.
To enhance safety, we recommend using granite-vision-3.2-2b alongside Granite Guardian. Granite Guardian is a fine-tuned instruct model designed to detect and flag risks in prompts and responses across key dimensions outlined in the IBM AI Risk Atlas. Its training, which includes both human-annotated and synthetic data informed by internal red-teaming, enables it to outperform similar open-source models on standard benchmarks, providing an additional layer of safety.

**Resources**
- 📄 Read the full technical report [here](https://arxiv.org/abs/2502.09927)
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 🚀 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources
