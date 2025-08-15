# Vision-Language Grounding Transformer (VL-GTR)

This repository contains the implementation of the Vision-Language Grounding Transformer (VL-GTR), a model designed to localize objects in images based on natural language descriptions. Given an image and a phrase like "the man in the red shirt," the model predicts the bounding box for the described object.

This project was developed to address the "Scene Localization in Dense Images via Natural Language Queries" problem statement.

---

## 🚀 Core Technologies

The model is built using a modern deep learning stack:

* **PyTorch:** The core deep learning framework.
* **timm (PyTorch Image Models):** Used for the high-performance Vision Transformer (ViT) backbone.
* **Hugging Face Transformers:** Used for the powerful RoBERTa text encoder.
* **Scipy, NumPy, Matplotlib, Pillow:** Essential libraries for data manipulation, scientific computing, and visualization.

---

## 🏛️ Model Architecture

The VL-GTR model is heavily inspired by the DETR (DEtection TRansformer) architecture, providing an end-to-end solution for visual grounding.

1.  **Vision Backbone:** We use a pre-trained **`vit_small_patch14_reg4_dinov2.lvd142m`** model from `timm`. DINOv2 models are known for their robust self-supervised features, making them excellent for downstream tasks like grounding.
2.  **Text Backbone:** The input phrase is encoded using **`roberta-base`** from Hugging Face. The final hidden state of the `[CLS]` token serves as the comprehensive embedding for the phrase.
3.  **Transformer & Prediction Heads:** The core of the model fuses the visual and textual information.
    * The image and text embeddings are projected into a shared dimension.
    * A standard Transformer decoder takes this combined memory and a set of learned **object queries** as input.
    * The decoder's output is passed to two prediction heads:
        * A **Class Head** that predicts "object" vs. "no-object".
        * A **Box Head** that predicts the bounding box coordinates (center_x, center_y, width, height).

---

## 📚 Dataset

The model was trained on the **Flickr30k Entities** dataset, which provides rich annotations linking phrases in sentences to their corresponding bounding boxes in images.

---

## ⚙️ Training Strategy

A key to the model's success was a two-stage training strategy to ensure stability and prevent catastrophic forgetting of the pre-trained weights.

1.  **Stage 1: Training the Heads:** Initially, the vision and text backbones are completely **frozen** (`requires_grad = False`). Only the newly initialized components (the Transformer, projection layers, and prediction heads) are trained. This allows the "glue" of the model to learn its task without corrupting the powerful pre-trained features.
2.  **Stage 2: Fine-Tuning the Vision Backbone:** Once the heads have stabilized, the vision backbone is **unfrozen**. A new optimizer with a much lower learning rate (e.g., `1e-5`) is used to make small, careful adjustments to the visual features, adapting them specifically for the grounding task.

---

## Usage

### 1. Setup

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/vl-gtr.git](https://github.com/your-username/vl-gtr.git)
cd vl-gtr
pip install -r requirements.txt
````

*(Note: You will need to create a `requirements.txt` file with libraries such as `torch`, `timm`, `transformers`, `scipy`, `numpy`, `matplotlib`, `Pillow`)*

### 2\. Data Preparation

Download the Flickr30k dataset and annotations. Update the paths in the configuration section of the notebook (`aims-project-1-model-2-8088a6.ipynb`) to point to your data directories.

### 3\. Training

The entire training process is detailed in the Jupyter Notebook `aims-project-1-model-2-8088a6.ipynb`. You can run the cells sequentially to parse the data, build the model, and execute the training loop. The best performing model will be saved as `vl_gtr_best_model.pth`.

### 4\. Inference and Visualization

To visualize predictions on new images, you can use the `visualize_prediction` function from the notebook. Load the saved model weights and provide the model with a sample from the dataset (or a custom-preprocessed sample).

```python
# Load the best model
model.load_state_dict(torch.load('vl_gtr_best_model.pth'))

# Run visualization
visualize_prediction(model, val_dataset, device='cuda', num_samples=5)
```

-----

## 📈 Results

The model achieves strong performance on the validation set, accurately localizing a wide variety of objects described by complex phrases.

*(This is where you would add images showing prediction examples, with green boxes for ground truth and red boxes for model predictions)*

**Example Output:**

  * **Phrase:** "A man in a white shirt"
  * **Result:** The model correctly draws a bounding box around the specified person.

-----

## 🔮 Future Improvements

  * **Train on Larger Datasets:** Improve generalization by training on more extensive datasets like Visual Genome or RefCOCO/RefCOCO+.
  * **Experiment with Advanced Backbones:** Test newer, more powerful models like ViT-Large or DeBERTa.
  * **Handle Multiple Objects:** Extend the architecture to ground multiple phrases from a single, more complex sentence.
  * **Model Optimization:** Optimize the model for faster inference using techniques like pruning, quantization, or conversion to ONNX/TensorRT for deployment.

## Kaggle direct link
https://www.kaggle.com/code/bhaveshbhardwaj7/aims-project-1-model-2

<!-- end list -->

```
```
