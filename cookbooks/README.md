Here we provide all aviable notebooks for using different capabilities of Granite Vision models.

### Fine-tuning

For an example of fine-tuning Granite Vision for new tasks refer to [this notebook](https://huggingface.co/learn/cookbook/en/fine_tuning_granite_vision_sft_trl).

### Use Granite Vision for MM RAG

For an example of MM RAG using granite vision refer to [this notebook](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/RAG/Granite_Multimodal_RAG.ipynb).

### Segmentation

For an example of referring segmentation using granite vision check out [this notebook](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/cookbooks/GraniteVision_Segmentation_Notebook.ipynb)

### Semgentation with optional Bounding Box creation

For an extended example of referring segmentation using granite vision that optionally derives a **bounding box** from the predicted mask for objection localization, check out [this notebook](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/cookbooks/GraniteVision_Segmentation_Notebook_add_Bbox.ipynb)

Read the quick start guide below for dependencies and notebok usage examples.
## Quick start

Open the notebook:

- `GraniteVision_Segmentation_Notebook_add_Bbox.ipynb`

### Install dependencies

If you're running locally (venv/conda), you can install requirements with:

```bash
pip install "transformers torch pillow requests
```

> **Note on PyTorch:** Depending on your OS/CUDA setup, you may need to install PyTorch using the official selector: https://pytorch.org/get-started/locally/

## What the notebook does

1. Loads an image (URL or local file)
2. **Preprocesses** the image (recommended): pad-to-square + resize
3. Runs the vision-language segmentation model with a text prompt (e.g. `"the leather handbag"`)
4. Converts the coarse model output into a full-resolution mask
5. Overlays the mask on the image
6. (Optional) Computes a bounding box from the mask for downstream tasks


### How big should the item be?

This model’s segmentation is **coarse** (24×24 grid), so it works best when the target item occupies roughly **25–35%+** of the frame.
If you’re segmenting items in full-body outfit photos, consider:
1) cropping around the item (using a detector or manual crop),
2) padding that crop to square,
3) then segmenting.

## Prompting tips for fashion items

Good prompts are **specific**:

- `"the brown leather handbag"`
- `"the blue car"`
- `"the copper cooking pot"`


If multiple candidates exist, add a locator:

- `"the red car on the right lane"`
- `"the dog laying on the left"`

### Doctags
