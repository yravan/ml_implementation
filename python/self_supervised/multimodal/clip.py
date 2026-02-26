"""
CLIP: Learning Transferable Models for Computer Vision via Contrastive Language-Image Pretraining

CLIP learns visual representations by contrasting images with their text descriptions.
The key innovation is learning from diverse image-text pairs from the web, enabling
robust zero-shot transfer without downstream fine-tuning.

Paper: "Learning Transferable Models For Computer Vision Tasks"
       https://arxiv.org/abs/2103.00020
       Radford et al. (OpenAI), 2021

Theory:
========
CLIP connects vision and language through contrastive learning:

Key Insight: Learning Aligned Vision-Language Representations
==============================================================

Traditional supervised learning:
  image → [CNN] → class_logits → cross_entropy(labels)
  Requires labeled data

Zero-shot learning (naive):
  Given new class not in training data, cannot classify
  Limited by training classes

CLIP approach:
  image → [Vision Encoder] → image_embedding
  text → [Language Encoder] → text_embedding
  Similarity: image_embedding · text_embedding

  Zero-shot: Compare image to {class_1, class_2, ...} embeddings

Key Properties:

1. **Language as Supervision**:
   - Instead of class labels, use natural language descriptions
   - "A dog" vs "A cat" vs "A bird"
   - Much more expressive than discrete labels

2. **Contrastive Objective**:
   - Learn from pairs: (image, text_description)
   - Maximize similarity between image and matching text
   - Minimize similarity with non-matching text
   - Similar to InfoNCE loss

3. **Scale of Training Data**:
   - Trained on 400 million image-text pairs from internet
   - Huge diversity in images, objects, scenes, text
   - Enables learning of general visual concepts

4. **Zero-Shot Transfer**:
   - No fine-tuning on downstream tasks
   - Use text to describe target classes
   - Classify by finding most similar text embedding
   - Remarkably effective

Architecture:
==============

Vision Encoder:
  Image (224×224, RGB)
         ↓
  [Vision Transformer or ResNet]
  Global Average Pool
         ↓
  image_embedding ∈ ℝ^512

Text Encoder:
  Text sequence (e.g., "a photograph of a dog")
         ↓
  [Transformer / LSTM]
  Attention pool (on [CLS] token or similar)
         ↓
  text_embedding ∈ ℝ^512

Both embeddings:
  - Project to same space
  - Normalized to unit sphere (L2 norm = 1)
  - Enable semantic similarity computation

Training Process:
=================

Data:
  - Image-text pairs from web (4B image-text pairs)
  - Diverse: objects, scenes, actions, concepts
  - Noisy: texts include alt-text, hashtags, captions

For each batch of N image-text pairs:

1. Encode all images: I = [i_1, ..., i_N] → [N, 512]
2. Encode all texts: T = [t_1, ..., t_N] → [N, 512]

3. Compute similarity matrix:
   S[i,j] = I[i] · T[j] / τ  (temperature-scaled dot product)
   Shape: [N, N]
   Diagonal elements are positive pairs (should be high)
   Off-diagonal are negative pairs (should be low)

4. Contrastive loss:
   - For each image i: softmax over similarities with all texts
   - Target: t_i (matching text for image i)
   - Loss: CrossEntropy(S[i,:], one_hot(i))
   - Same for text: each text matches its image

5. Symmetric loss:
   - Image→Text loss: classify text given image
   - Text→Image loss: classify image given text
   - L_total = (L_image2text + L_text2image) / 2

Mathematical Formulation:
=========================

Let:
  - I_i = image embedding (normalized)
  - T_j = text embedding (normalized)
  - τ = temperature parameter

Logits (per-image cross-entropy):
  logits_i = (I_i @ T^T) / τ  [logits for image i with all texts]

Loss (contrastive):
  L_i = CrossEntropy(logits_i, one_hot(i))
      = -log[exp(I_i · T_i / τ) / Σ_j exp(I_i · T_j / τ)]

Symmetric Loss:
  L = (Σ_i L_i^(image→text) + Σ_i L_i^(text→image)) / (2N)

Key hyperparameters:
  - Temperature τ ≈ 0.07 (empirically optimal)
  - Batch size: 32768 (huge batches help)
  - Learning rate: Linear warmup then cosine decay

Why CLIP Works:
================

1. **Language as Soft Labels**:
   - "A dog" more informative than class_id = 0
   - Captures multiple concepts simultaneously
   - Model learns compositional understanding

2. **Natural Diversity**:
   - Training data from internet (diverse)
   - Not restricted to clean labeled datasets
   - Covers edge cases, uncommon concepts

3. **Emergent Zero-Shot Ability**:
   - Not explicitly trained for zero-shot
   - Emerges from learning image-text alignment
   - "A dog" text never seen during training
   - But composition learned from "dog" + "photo" + etc.

4. **Scaling Enables Generalization**:
   - Larger models: Better performance
   - More diverse data: Better transfer
   - Shows self-supervised scaling benefits

Zero-Shot Transfer:
====================

Classical supervised classification:
  Input image → Pre-trained model → Features → Linear classifier (trained) → Class

CLIP zero-shot:
  1. Encode class descriptions as text:
     class_0_text = "A photo of a dog"
     class_1_text = "A photo of a cat"
     ...
     class_embeddings = [encode(t) for t in class_texts]

  2. Encode test image:
     image_embedding = encode(image)

  3. Compute similarity with all class embeddings:
     similarities = image_embedding @ class_embeddings^T

  4. Predict:
     predicted_class = argmax(similarities)

Example: ImageNet Zero-Shot
  - 1000 classes in ImageNet
  - Create text for each class: "A photo of [class]"
  - Encode all 1000 texts
  - For each test image, find most similar text
  - Accuracy: ~76% (comparable to supervised!)

Prompt Engineering:
===================

Text representations vary with wording:

"A dog" vs "A photo of a dog" vs "A portrait of a dog"
→ Different text embeddings!

CLIP performance depends on prompt quality:

Bad prompt: "dog"
  - Too ambiguous
  - Might not capture visual features

Good prompt: "A photo of a dog"
  - More specific
  - Includes context (photo)

Prompt templates help:
  - "A photo of a {class}"
  - "A {class} in a scene"
  - "The texture of {class}"
  - Ensemble multiple prompts for better performance

Why this matters:
  - Same image, different prompts → different predictions
  - Prompts encode domain knowledge
  - Example: "A dog breed: {class}" better for breed classification

Downstream Evaluation:
======================

1. **Zero-Shot Classification**:
   - ImageNet: 76% accuracy (vs 81% supervised)
   - CIFAR-10: 95.4% (vs 99% supervised)
   - Remarkable without any downstream training!

2. **Few-Shot Learning**:
   - With just 1-4 examples per class
   - Can fine-tune linear probe on frozen features
   - Often better than full fine-tuning

3. **Linear Probing**:
   - Freeze CLIP encoder
   - Train linear classifier on downstream task
   - Usually 80-90% of fine-tuning performance

4. **Transfer Learning**:
   - Fine-tune entire model on downstream task
   - Better performance than linear probing
   - Still benefits from CLIP initialization

Advantages of CLIP:
===================
  + Enables zero-shot transfer without downstream training
  + Learns from diverse, natural image-text pairs
  + Remarkably robust to distribution shift
  + Few-shot learning very effective
  + Interpretable via natural language
  + Scales well with data and model size
  + No need for expensive labeled datasets

Disadvantages of CLIP:
=====================
  - Requires huge amounts of image-text data (expensive to collect)
  - Training expensive (huge batch sizes needed)
  - Text descriptions sometimes not ideal
  - Distribution shift: still performs worse on out-of-distribution
  - Societal bias: trained on internet data (contains biases)

Robustness Properties:
======================

CLIP shows interesting robustness properties:

1. **Distribution Shift Robustness**:
   - ImageNet → ImageNet-A,V,R: Smaller drop than supervised
   - Better OOD robustness than supervised learning
   - Diversity of web data helps

2. **Adversarial Robustness**:
   - More robust to adversarial examples
   - Language grounding provides regularization

3. **Bias and Fairness**:
   - Inherits biases from training data (internet)
   - But can use prompts to mitigate some biases
   - Example: "A photo of a CEO" sometimes correlates with gender
   - Prompts: "A female CEO", "A male CEO" can balance

Theoretical Understanding:
==========================

Why Language Supervision is Powerful:

1. **Implicit Regularization**:
   - Learning from text provides implicit constraint
   - Images must align with semantic meaning
   - Prevents trivial solutions (collapse)

2. **Compositional Learning**:
   - Learned by composing concepts from text
   - "black dog" = "black" + "dog"
   - Enables generalization to unseen combinations

3. **Information Bottleneck**:
   - Images and text must fit through embedding
   - Forces learning of essential structure
   - Language provides semantic guidance

Scaling Laws:
==============

Effect of Model Size (on zero-shot accuracy):

  ViT-B (86M): 63% ImageNet
  ViT-L (303M): 75%
  ViT-g (1.8B): 80%

Effect of Training Data Size:

  400M pairs: 70% ImageNet
  1B pairs: 76%
  Multiple datasets: 80%+

Shows consistent scaling benefits!

Related Work and Variants:
==========================

Contrastive Language-Image Models:

1. **ALIGN**:
   - Earlier approach (Google)
   - Similar to CLIP but with slight differences
   - Slightly different architecture

2. **BLIP**:
   - Adds caption generation
   - More modalities (Q&A, etc.)
   - Improves over CLIP

3. **Flamingo**:
   - Multimodal few-shot learner
   - Builds on CLIP concepts
   - Adds visual reasoning

4. **LLaVA**:
   - Vision language model
   - Combines ViT with LLM
   - Uses CLIP-like training

Implementation Considerations:
============================

1. **Huge Batch Sizes**:
   - Need large global batch size (32k+)
   - Requires distributed training
   - Critical for good performance
   - Can use gradient accumulation

2. **Temperature Scaling**:
   - τ ≈ 0.07 (learnable parameter)
   - Controls softness of contrastive loss
   - Important for convergence

3. **Symmetric Loss**:
   - Must compute both image→text and text→image
   - Equal contribution
   - Ensures both modalities learned equally

4. **Prompt Engineering**:
   - Text descriptions matter
   - Better prompts → better zero-shot
   - Multiple prompts can be ensembled

5. **Computational Requirements**:
   - Huge training cost (weeks on multiple V100s)
   - Data cost: 400M image-text pairs
   - Not practical for small labs
   - However: Pre-trained models available for use

Practical Usage:
===============

Most practitioners use pre-trained CLIP:

1. Load model:
   import clip
   model, preprocess = clip.load("ViT-B/32")

2. Zero-shot classification:
   classes = ["dog", "cat", "bird"]
   texts = clip.tokenize([f"a photo of a {c}" for c in classes])
   text_features = model.encode_text(texts)
   image = preprocess(pil_image).unsqueeze(0)
   image_features = model.encode_image(image)
   similarities = image_features @ text_features.T

3. This enables instant zero-shot classification!

Why CLIP Changed ML:
====================

1. **Paradigm Shift**:
   - From supervised learning (labeled data)
   - To zero-shot learning (natural language)
   - Enables AI systems without downstream training

2. **Scaling Insight**:
   - Large amounts of unlabeled data + learning signal
   - Better than small supervised datasets
   - Information theory suggests this

3. **Practical Impact**:
   - Enables fast deployment to new tasks
   - No need for task-specific annotations
   - Foundation for many subsequent models

4. **Research Direction**:
   - Showed promise of multimodal learning
   - Inspired many follow-up works
   - Foundation for modern vision-language models

Future Directions:
==================

1. **Video Understanding**:
   - Extend CLIP to video-text
   - Temporal understanding
   - Action recognition

2. **Reasoning**:
   - Beyond image classification
   - Visual reasoning, counting, etc.
   - Combine with language models

3. **Efficiency**:
   - Smaller CLIP models
   - Distillation approaches
   - Faster inference

4. **Robustness**:
   - Better handling of distribution shift
   - Adversarial robustness
   - Fairness and bias mitigation

5. **Multimodality**:
   - Extend beyond image-text
   - Audio, 3D, video
   - Joint embeddings
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from python.nn_core import Module, Parameter


class ImageEncoder(Module):
    """
    Vision encoder for CLIP.

    Encodes images to embeddings.

    Typical architectures:
    - Vision Transformer (ViT): ViT-B/32, ViT-L/14
    - ResNet: ResNet50, ResNet101

    Output: Normalized embeddings [batch_size, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        architecture: str = 'resnet50'
    ):
        """
        Args:
            embedding_dim: Output embedding dimension
            architecture: 'resnet50', 'resnet101', or 'vit-b'
        """
        super().__init__()
        raise NotImplementedError(
            "Implement image encoder:\n"
            "1. Load base architecture (ResNet or ViT)\n"
            "2. Remove classification head\n"
            "3. Add projection head to embedding_dim\n"
            "4. Normalize output to unit sphere\n"
            "Hint: Use torchvision models or timm library"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Encode image to embedding.

        Args:
            x: Input image [batch_size, 3, 224, 224]

        Returns:
            Normalized embedding [batch_size, embedding_dim]
        """
        raise NotImplementedError()


class TextEncoder(Module):
    """
    Text encoder for CLIP.

    Encodes text descriptions to embeddings.

    Typical architecture:
    - Transformer (12-24 blocks)
    - Output: Pooled embedding from special token

    Output: Normalized embeddings [batch_size, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        vocab_size: int = 49408,
        context_length: int = 77,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12
    ):
        """
        Args:
            embedding_dim: Output embedding dimension
            vocab_size: Size of vocabulary
            context_length: Maximum sequence length
            transformer_width: Width of transformer
            transformer_heads: Number of attention heads
            transformer_layers: Number of transformer layers
        """
        super().__init__()
        raise NotImplementedError(
            "Implement text encoder:\n"
            "1. Token embedding layer\n"
            "2. Positional embedding layer\n"
            "3. Stack of Transformer blocks\n"
            "4. LayerNorm\n"
            "5. Projection to embedding_dim\n"
            "Note: Use <|endoftext|> token for pooling"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Encode text to embedding.

        Args:
            x: Tokenized text [batch_size, context_length]

        Returns:
            Normalized embedding [batch_size, embedding_dim]
        """
        raise NotImplementedError()


class CLIPModel(Module):
    """
    CLIP model combining vision and language encoders.

    Usage:
        model = CLIPModel()
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        logits = image_features @ text_features.T
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        vision_model: str = 'resnet50',
        text_vocab_size: int = 49408,
        temperature: float = 0.07
    ):
        """
        Args:
            embedding_dim: Shared embedding dimension
            vision_model: Vision model architecture
            text_vocab_size: Vocabulary size for tokenizer
            temperature: Softmax temperature (learnable)
        """
        super().__init__()
        raise NotImplementedError(
            "Implement CLIP model:\n"
            "1. Create image encoder\n"
            "2. Create text encoder\n"
            "3. Create learnable temperature parameter\n"
            "4. Store configuration"
        )

    def encode_image(self, x: np.ndarray) -> np.ndarray:
        """Encode image to normalized embedding."""
        raise NotImplementedError()

    def encode_text(self, x: np.ndarray) -> np.ndarray:
        """Encode text to normalized embedding."""
        raise NotImplementedError()

    def forward(
        self,
        images: np.ndarray,
        texts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for contrastive learning.

        Args:
            images: Image batch [batch_size, 3, 224, 224]
            texts: Tokenized text [batch_size, seq_length]

        Returns:
            image_features: [batch_size, embedding_dim]
            text_features: [batch_size, embedding_dim]
            temperature: Learned temperature
        """
        raise NotImplementedError()


class CLIPLoss(Module):
    """
    Contrastive loss for CLIP training.

    Implements symmetric loss:
    - Image → Text: Classify text given image
    - Text → Image: Classify image given text

    L = (L_i2t + L_t2i) / 2

    Where L_i2t is cross-entropy between image-text similarities and one-hot.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        temperature: float
    ) -> float:
        """
        Compute symmetric contrastive loss.

        Args:
            image_features: [batch_size, embedding_dim]
            text_features: [batch_size, embedding_dim]
            temperature: Scalar temperature value

        Returns:
            Scalar loss value

        Implementation:
        1. Normalize both feature sets (already done in encoder)
        2. Compute logits: logits = image_features @ text_features.T / temperature
           Shape: [batch_size, batch_size]
        3. Create labels: one-hot with 1 at diagonal (matching pairs)
        4. Image→Text loss: classify text given image
           loss_i2t = CrossEntropy(logits, labels)
        5. Text→Image loss: classify image given text
           loss_t2i = CrossEntropy(logits.T, labels)
        6. Total: loss = (loss_i2t + loss_t2i) / 2
        """
        raise NotImplementedError(
            "Implement CLIP loss:\n"
            "1. logits = (image_features @ text_features.T) / temperature\n"
            "2. batch_size = logits.shape[0]\n"
            "3. labels = torch.arange(batch_size)  # Diagonal matches\n"
            "4. loss_i2t = self.criterion(logits, labels)\n"
            "5. loss_t2i = self.criterion(logits.T, labels)\n"
            "6. loss = (loss_i2t + loss_t2i) / 2\n"
            "7. Return loss"
        )


class CLIPDataset:
    """
    Dataset for CLIP training.

    Assumes input is list of (image_path, text_description) pairs.

    Example:
        data = [
            ("image1.jpg", "a dog in the park"),
            ("image2.jpg", "a cat on the table"),
        ]
        dataset = CLIPDataset(data, image_transform, tokenizer)
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        image_transform,
        tokenizer,
        context_length: int = 77
    ):
        """
        Args:
            image_paths: List of image file paths
            captions: List of text captions (one per image)
            image_transform: Image preprocessing function
            tokenizer: Text tokenization function
            context_length: Max length for tokenized text
        """
        self.image_paths = image_paths
        self.captions = captions
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Returns:
            {
                'image': preprocessed image,
                'text': tokenized text
            }

        Implementation:
        1. Load image from path
        2. Apply image transformation
        3. Tokenize caption to fixed length
        4. Return dict with both
        """
        raise NotImplementedError(
            "Implement __getitem__:\n"
            "1. image = Image.open(self.image_paths[idx])\n"
            "2. image = self.image_transform(image)\n"
            "3. text = self.tokenizer(self.captions[idx], context_length)\n"
            "4. Return {'image': image, 'text': text}"
        )


class CLIPTrainer:
    """
    Trainer for CLIP training.

    Handles:
    - Loading image-text pairs
    - Computing losses
    - Distributed training support
    - Checkpoint saving/loading

    Note: CLIP requires huge batch sizes (32k+) for good performance.
    This typically requires multi-GPU training.

    Usage:
        model = CLIPModel()
        trainer = CLIPTrainer(model, train_loader, device='cuda')
        for epoch in range(epochs):
            train_loss = trainer.train_epoch()
    """

    def __init__(
        self,
        model: CLIPModel,
        optimizer,
        train_loader,
        loss_fn: CLIPLoss,
        device: str = 'cpu',
        world_size: int = 1,
        rank: int = 0
    ):
        """
        Args:
            model: CLIPModel instance
            optimizer: Optimizer (SGD or AdamW)
            train_loader: Training data loader
            loss_fn: CLIPLoss instance
            device: 'cpu' (no GPU support in custom Module system)
            world_size: Number of GPUs for distributed training
            rank: Current GPU rank
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.device = device
        self.world_size = world_size
        self.rank = rank

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch:
           a. Load images and texts
           b. Forward pass: image_features, text_features, temp = model(images, texts)
           c. Compute loss: loss = loss_fn(image_features, text_features, temp)
           d. Backward and optimizer step
           e. Track running loss
        3. Return average loss

        Important Notes:
        - Requires all-gather for multi-GPU training
        - Global batch size must be huge (32k+)
        - Can use gradient accumulation if needed
        """
        raise NotImplementedError(
            "Implement training loop:\n"
            "1. self.model.train()\n"
            "2. For each batch:\n"
            "   a. images, texts = batch\n"
            "   b. image_feat, text_feat, temp = self.model(images, texts)\n"
            "   c. loss = self.loss_fn(image_feat, text_feat, temp)\n"
            "   d. loss.backward()\n"
            "   e. self.optimizer.step()\n"
            "   f. Track loss\n"
            "3. Return average loss"
        )

    def evaluate(self) -> float:
        """Evaluate on validation set."""
        raise NotImplementedError()


class ZeroShotClassifier:
    """
    Classifier using CLIP for zero-shot prediction.

    Usage:
        classifier = ZeroShotClassifier(clip_model, device='cuda')
        classes = ['dog', 'cat', 'bird']
        logits = classifier(images, classes)
        predictions = logits.argmax(dim=1)
    """

    def __init__(
        self,
        model: CLIPModel,
        tokenizer,
        device: str = 'cuda',
        prompt_template: str = "a photo of a {}"
    ):
        """
        Args:
            model: Trained CLIPModel
            tokenizer: Text tokenizer
            device: Device to use
            prompt_template: Template for generating class prompts
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_template = prompt_template

    def __call__(
        self,
        images: np.ndarray,
        classes: List[str]
    ) -> np.ndarray:
        """
        Classify images using text-based class descriptions.

        Args:
            images: Image batch [batch_size, 3, 224, 224]
            classes: List of class names

        Returns:
            Logits [batch_size, num_classes]

        Implementation:
        1. Generate prompts: prompts = [template.format(c) for c in classes]
        2. Tokenize: text_tokens = [tokenizer(p) for p in prompts]
        3. Encode text: text_features = model.encode_text(text_tokens)
        4. Encode images: image_features = model.encode_image(images)
        5. Compute similarities: logits = image_features @ text_features.T
        6. Return logits
        """
        raise NotImplementedError(
            "Implement zero-shot classification:\n"
            "1. prompts = [self.prompt_template.format(c) for c in classes]\n"
            "2. text_tokens = torch.cat([self.tokenizer(p) for p in prompts])\n"
            "3. with torch.no_grad():\n"
            "     text_features = self.model.encode_text(text_tokens)\n"
            "     image_features = self.model.encode_image(images)\n"
            "4. logits = image_features @ text_features.T / temperature\n"
            "5. Return logits"
        )


# ============================================================================
# Understanding CLIP
# ============================================================================

"""
Key Insights about CLIP:

1. **Language as Implicit Labels**:
   - Natural language more expressive than class IDs
   - Encodes rich semantic information
   - Model learns compositional understanding

2. **Scaling and Data**:
   - More data (400M pairs) > More labeled examples (1M)
   - Diversity of web data crucial
   - Shows self-supervised learning advantages

3. **Emergent Zero-Shot Ability**:
   - Not explicitly trained for zero-shot
   - Emerges from image-text alignment
   - Remarkable generalization to new classes

4. **Robustness to Distribution Shift**:
   - Better OOD performance than supervised
   - Language grounding provides regularization
   - Diversity of web data helps

5. **Scalability**:
   - Larger models = better performance
   - Different from supervised (plateau)
   - Self-supervised benefits

Impact on AI Research:
  - Shifted focus from supervised to vision-language
  - Inspired many multimodal models
  - Showed promise of language-guided vision
  - Foundation for modern multimodal AI

Future Work:
  - Video-language models
  - Reasoning and compositionality
  - Efficiency and deployment
  - Fairness and bias mitigation
"""
