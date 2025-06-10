# granite-vision-3.3-2b-embedding
**Model Summary:**
Granite-vision-3.3-2b-embedding is an efficient embedding model based on granite-vision-3.3-2b. This model is specifically designed for multimodal document retrieval, enabling queries on documents with tables, charts, infographics, and complex layout. The model generates ColBERT-style multi-vector representations of pages.
By removing the need for OCR-based text extractions, granite-vision-3.3-2b-embedding can help simplify and accelerate RAG pipelines.

**Evaluations:**
We evaluated granite-vision-3.3-2b-embedding alongside other top colBERT style multi-modal embedding models in the 1B-4B parameter range using two benchmark: Vidore2 and [Real-MM-RAG-Bench](https://arxiv.org/abs/2502.12342) which aim to specifically address complex multimodal document retrieval tasks.

## **NDCG@5 - ViDoRe V2**
| Collection \ Model                     | ColPali-v1.3 | ColQwen2.5-v0.2 | ColNomic-3b |  ColSmolvlm-v0.1     |  granite-vision-3.3-2b-embedding |
|----------------------------------------|--------------|------------------|-------------|-------------------|-----------
| ESG Restaurant Human                   | 51.1        | 68.4           | 65.8       |    62.4               | 62.3                    |
| Economics Macro Multilingual           | 49.9        | 56.5            | 55.4       |     47.4              | 48.3                    |
| MIT Biomedical                         | 59.7        | 63.6            | 63.5       |    58.1               |60.0                   |
| ESG Restaurant Synthetic               | 57.0        | 57.4            | 56.6       |     51.1              |54.0                    |
| ESG Restaurant Synthetic Multilingual  | 55.7        | 57.4            | 57.2       |     47.6             |53.5                    |
| MIT Biomedical Multilingual            | 56.5        | 61.1            | 62.5       |      50.5             | 53.6                    |
| Economics Macro                        | 51.6        | 59.8            | 60.2       |      60.9            |60.0                    |
| **Avg (ViDoRe2)**                      | **54.5**    | **60.6**        | **60.2**   | **54.0**              |**56.0**                    |

## **NDCG@5 - REAL-MM-RAG**
| Collection \ Model                     | ColPali-v1.3 | ColQwen2.5-v0.2 | ColNomic-3b |   ColSmolvlm-v0.1            |  granite-vision-3.3-2b-embedding |
|----------------------------------------|--------------|------------------|-------------|--------------------------| ------------------
| FinReport                              | 55         | 66             | 78        |   65                  |70 
| FinSlides                              | 68        | 79             | 81        |   55                 |74  
| TechReport                             | 78         | 86             | 88        |   83                 |84  
| TechSlides                             | 90         | 93             | 92        |   91            |93   
| **Avg (REAL-MM-RAG)**                  | **73**     | **81**         | **85**    |   **74**           |**80**    

- **Release Date**: June 11th 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Supported Input Format:**
Currently the model supports English queries and images (png, jpeg, etc.) as input format.

**Intended Use:**
The model is intended to be used in enterprise applications that involve retrieval of visual and text data. In particular, the model is well-suited for multi-modal RAG systems where the knowledge base is composed of complex enterprise documents, such as reports, slides, images, canned doscuments, manuals and more. The model can be used as a standalone retriever, or alongside a text-based retriever.

### Usage
```shell
pip install -q torch torchvision torchaudio
pip install transformers==4.50
```
Then run the code:
```python
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ibm-granite/granite-vision-3.3-2b-embedding"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# ─────────────────────────────────────────────
# Inputs: Image + Text
# ─────────────────────────────────────────────
image_url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
print("\nFetching image...")
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

text = "A photo of a tiger"
print(f"Image and text inputs ready.")

# Process both inputs
print("Processing inputs...")
image_inputs = processor.process_images([image])
text_inputs = processor.process_queries([text])

# Move to correct device
image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

# ─────────────────────────────────────────────
# Run Inference
# ─────────────────────────────────────────────
with torch.no_grad():
    print("🔍 Getting image embedding...")
    img_emb = model(**image_inputs)

    print("✍️ Getting text embedding...")
    txt_emb = model(**text_inputs)

# ─────────────────────────────────────────────
# Score the similarity
# ─────────────────────────────────────────────
print("Scoring similarity...")
similarity = processor.score(txt_emb, img_emb, batch_size=1, device=device)

print("\n" + "=" * 50)
print(f"📊 Similarity between image and text: {similarity.item():.4f}")
print("=" * 50)
```
### Use granite-vision-embedding-3.3-2b for MM RAG
For an example of MM-RAG using granite-vision-3.3-2b-embedding refer to [this notebook](......).

**Model Architecture:**
The architecture of granite-vision-3.3-2b-embedding follows ColPali(https://arxiv.org/abs/2407.01449) approach and consists of the following components:

(1) Vision-Language model : granite-vision-3.3-2b (https://huggingface.co/ibm-granite/granite-vision-3.3-2b).

(2) Projection layer: linear layer that projects the hidden layer dimension of Vision-Language model to 128 and outputs 729 embedding vectors per image.

The scoring is computed using MaxSim-based late interaction mechanism.

**Training Data:**
Our training data is entirly comprised from DocFM. DocFM is a large-scale comprehensive dataset effort at IBM consisting of 85 million document pages extracted from unique PDF
documents sourced from Common Crawl, Wikipedia, and ESG (Environmental, Social, and Governance)
reports.

**Infrastructure:**
We train granite-vision-3.3-2b-embedding on IBM’s cognitive computing cluster, which is outfitted with NVIDIA A100 GPUs.

**Ethical Considerations and Limitations:**
The use of Large Vision and Language Models involves risks and ethical considerations people must be aware of, including but not limited to: bias and fairness, misinformation, and autonomous decision-making. Granite-vision-3.3-2b-embedding is not the exception in this regard. Although our alignment processes include safety considerations, the model may in some cases produce inaccurate or biased responses.
Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use granite-vision-3.3-2b-embedding with ethical intentions and in a responsible way.

**Resources**
- 📄 Granite Vision technical report [here](https://arxiv.org/abs/2502.09927)
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 🚀 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources
