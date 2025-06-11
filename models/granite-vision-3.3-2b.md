# granite-vision-3.3-2b

**Model Summary:** Granite-vision-3.3-2b is a compact and efficient vision-language model, specifically designed for visual document understanding, enabling automated content extraction from tables, charts, infographics, plots, diagrams, and more. The model was trained on a meticulously curated instruction-following data, comprising diverse public and synthetic datasets tailored to support a wide range of document understanding and general image tasks. Granite-vision-3.3-2b was trained by fine-tuning a Granite large language model with both image and text modalities.


**Evaluations:** We compare the performance of granite-vision-3.3-2b with previous versions of granite-vision models. Evaluations were done using the standard llms-eval benchmark and spanned multiple public benchmarks, with particular emphasis on document understanding tasks while also including general visual question-answering benchmarks. 

| | GV-3.1-2b-preview | GV-3.2-2b | GV-3.3-2b |
|-----------|-----------|--------------|----------------|
| **Document benchmarks** |
| ChartQA | 0.86 | 0.87 | 0.87 |
| DocVQA | 0.88 | 0.89 | **0.91** |
| TextVQA | 0.76 | 0.78 | **0.80** |
| AI2D | 0.78 | 0.76 | 0.77 |
| InfoVQA | 0.63 | 0.64  | **0.68** |
| OCRBench | 0.75 | 0.77 | **0.79** |
| LiveXiv VQA v2 | 0.61 | 0.61 | 0.61 |
| LiveXiv TQA v2 | 0.55 | 0.57 | 0.52 |
| **Other benchmarks** |
| MMMU | 0.35 | 0.37 | 0.37 |
| VQAv2 | 0.81 | 0.78 | 0.79 |
| RealWorldQA | 0.65 | 0.63 | 0.63 |
| VizWiz VQA | 0.64 | 0.63 |  0.62 |
| OK VQA | 0.57 | 0.56 | 0.55|

- **Paper:** [Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence](https://arxiv.org/abs/2502.09927). Note that the paper describes Granite Vision 3.2. Granite Vision 3.3 shares most of the technical underpinnings with Granite 3.2. However, there are several enhancements in terms of new and improved vision encoder, many new high quality datasets for training, and several new experimental capabilities.
- **Release Date**: Jun 11th, 2025 
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Input Format:** Currently the model supports English instructions and images (png, jpeg) as input format. 

**Intended Use:**  The model is intended to be used in enterprise applications that involve processing visual and text data. In particular, the model is well-suited for a range of visual document understanding tasks, such as analyzing tables and charts, performing optical character recognition (OCR), and answering questions based on document content. Additionally, its capabilities extend to general image understanding, enabling it to be applied to a broader range of business applications. For tasks that exclusively involve text-based input, we suggest using our Granite large language models, which are optimized for text-only processing and offer superior performance compared to this model.


## Generation:

Granite Vision model is supported natively `transformers>=4.49`. Below is a simple example of how to use the `granite-vision-3.3-2b` model.

### Usage with `transformers`

First, make sure to build the latest versions of transformers:
```shell
pip install transformers>=4.49
```

Then run the code:
```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.3-2b"
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

model_path = "ibm-granite/granite-vision-3.3-2b"

model = LLM(
    model=model_path,
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



### Safety evaluation

The GV-3.3-2b model also went through safety alignment to make sure responses are safer without affecting the model’s performance on its intended task. We carefully safety aligned the model on publicly available safety data and synthetically generated safety data. We report our safety scores on publicly available RTVLM and VLGuard datasets.

**RTVLM Safety Score - [0,10] - Higher is Better**

| | Politics | Racial | Jailbreak | Mislead |
|-----------|-----------|--------------|----------------|----------------|
|GV-3.1-2b-preview|7.2|7.7|4.5|7.6|
|GV-3.2-2b|7.6|7.8|6.2|8.0|
|GV-3.3-2b|8.0|8.1|7.5|8.0|


**VLGuard Safety Score - [0,10] - Higher is Better**

| | Unsafe Images (Unsafe) | Safe Images with Unsafe Instructions |
|-----------|-----------|--------------|
|GV-3.1-2b-preview|6.6|8.4|
|GV-3.2-2b|7.6|8.9|
|GV-3.3-2b|8.4|9.3|


### Experimental capabilities

Granite-vision-3.3-2b introduces three new experimental capabilities: 

(1) Image segmentation: [A notebook showing a segmentation example](https://github.com/ibm-granite/granite-vision-models/blob/main/cookbooks/GraniteVision_Segmentation_Notebook.ipynb)

(2) Doctags generation: Please see [Docling project](https://github.com/docling-project/docling) for more details on doctags.

(3) Multipage support: The model was trained to handle question answering (QA) tasks using multiple consecutive pages from a document—up to 10 pages—given the demands of long-context processing. To support such long sequences without exceeding GPU memory limits, we recommend resizing images so that their longer dimension is 768 pixels.


### Fine-tuning

For an example of fine-tuning granite-vision-3.3-2b for new tasks refer to [this notebook](https://huggingface.co/learn/cookbook/en/fine_tuning_granite_vision_sft_trl).

### Use Granite Vision for MM-RAG

For an example of MM-RAG using granite vision refer to [this notebook](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/RAG/Granite_Multimodal_RAG.ipynb).

**Model Architecture:**  The architecture of granite-vision-3.3-2b consists of the following components:

(1) Vision encoder: SigLIP2 (https://huggingface.co/google/siglip2-so400m-patch14-384).

(2) Vision-language connector: two-layer MLP with gelu activation function.

(3) Large language model: granite-3.1-2b-instruct with 128k context length (https://huggingface.co/ibm-granite/granite-3.1-2b-instruct).

We built upon LLaVA (https://llava-vl.github.io) to train our model. We use multi-layer encoder features and a denser grid resolution in AnyRes to enhance the model's ability to understand nuanced visual content, which is essential for accurately interpreting document images. 


**Training Data:**  Our training data is largely comprised of two key sources: (1) publicly available datasets (2) internally created synthetic data targeting specific capabilities including document understanding tasks. Granite Vision 3.3 training data is built upon the comprehensive dataset used for granite-vision-3.2-2b (a detailed description of granite-vision-3.2-2b training data is available in the [technical report](https://arxiv.org/abs/2502.09927)). In addition, granite-vision-3.3-2b further includes high quality image segmentation data, multi-page data, and data from several new high quality publicly available datasets (like Mammoth-12M and Bigdocs).  


**Infrastructure:** We train granite-vision-3.3-2b using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Responsible Use and Limitations:** Some use cases for  Large Vision and Language Models can trigger certain risks and ethical considerations, including but not limited to: bias and fairness, misinformation, and autonomous decision-making. Although our alignment processes include safety considerations, the model may in some cases produce inaccurate, biased, offensive or unwanted responses to user prompts. Additionally, whether smaller models may exhibit increased susceptibility to hallucination in generation scenarios due to their reduced sizes, which could limit their ability to generate coherent and contextually accurate responses, remains uncertain. This aspect is currently an active area of research, and we anticipate more rigorous exploration, comprehension, and mitigations in this domain. We urge the community to use granite-vision-3.3-2b in a responsible way and avoid any malicious utilization. We recommend using this model for document understanding tasks. More general vision tasks may pose higher inherent risks of triggering unwanted output. To enhance safety, we recommend using granite-vision-3.3-2b alongside Granite Guardian. Granite Guardian is a fine-tuned instruct model designed to detect and flag risks in prompts and responses across key dimensions outlined in the IBM AI Risk Atlas. Its training, which includes both human-annotated and synthetic data informed by internal red-teaming, enables it to outperform similar open-source models on standard benchmarks, providing an additional layer of safety.


**Resources**
- 📄 Read the full technical report [here](https://arxiv.org/abs/2502.09927)
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 🚀 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources
