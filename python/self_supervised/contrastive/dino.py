"""
DINO: Emerging Properties in Self-Supervised Vision Transformers

DINO (self-Distillation with No Labels) combines self-distillation and
multi-crop augmentations to learn rich visual representations without any labels.
Key innovation: Vision Transformers naturally learn semantic features through
self-supervised learning.

Paper: "Emerging Properties in Self-Supervised Vision Transformers"
       https://arxiv.org/abs/2104.14294
       Caron et al. (Facebook AI Research), 2021

Theory:
========
DINO introduces a knowledge distillation approach where:
1. Student network learns from teacher network
2. Teacher is updated via momentum (similar to BYOL/MoCo)
3. Both networks trained on multiple crops of same image
4. Loss: Cross-entropy between student and teacher outputs (softmax predictions)

Key Innovation: Self-Distillation Framework
=============================================

Unlike supervised knowledge distillation (teacher pre-trained, student learns):
- Both teacher and student learn from scratch
- Teacher is older version of student (momentum update)
- No labels required
- Applied to Vision Transformers

Architecture:
==============

Student Network:
  Image crop → [ViT Backbone] → [Linear Head] → [Softmax] → p_student

Teacher Network (momentum updated):
  Image crop → [ViT Backbone] → [Linear Head] → [Softmax] → p_teacher

Loss: Cross-entropy between student and teacher predictions
  L = -Σ_i p_teacher[i] * log(p_student[i])

Multi-Crop Strategy:
====================

Key difference from SimCLR/BYOL: Use multiple crops per image

Global Crops (2):
  - Large crops (224×224) covering most of image
  - Similar to standard augmentations

Local Crops (8):
  - Small crops (96×96) covering image details
  - Force network to learn local features
  - Student sees all global + local crops
  - Teacher sees only global crops

This asymmetry prevents trivial solutions:
  - Student must recognize local features
  - Teacher provides global context
  - Combination forces meaningful representations

Why Vision Transformers?
========================

Vision Transformers naturally suited for DINO:
1. **Self-attention**: Learns which parts of image are important
   - Attention maps show semantic object discovery
   - Without explicit supervision, ViT identifies objects

2. **Global receptive field**: Every token sees every pixel
   - Unlike CNNs which have limited local receptive field
   - Allows global reasoning from start

3. **[CLS] token**: Aggregates image information
   - Single token represents entire image
   - Used for downstream classification
   - Naturally emergent representation

4. **Scaling benefits**: ViTs scale well with data
   - Larger models learn better representations
   - Self-supervised learning utilizes this well

Emergent Properties:
====================

DINO training reveals:
1. **Object discovery**: Attention maps identify objects without bounding boxes
2. **Saliency**: Model learns what parts of image are important
3. **Segmentation**: Attention can be used for unsupervised segmentation
4. **Clustering**: Representations naturally cluster by semantic class

These properties emerge WITHOUT any supervision!

DINO Loss and Objective:
=========================

Cross-entropy loss with temperature scaling:

L = -Σ crop_i Σ crop_j≠i log[exp(sim(z_s^i, p_t^j) / τ_s) / Z]

Where:
  - z_s^i: Student output on crop i (un-normalized logits)
  - p_t^j: Teacher prediction on crop j (soft labels, normalized)
  - τ_s: Student temperature (lower value, sharper)
  - τ_t: Teacher temperature (higher value, softer)
  - Z: Partition function (normalization)

Teacher predictions are softmax outputs (soft labels):
  p_t = softmax(z_t / τ_t)

Student computes cross-entropy:
  L = CrossEntropy(softmax(z_s / τ_s), p_t)

Key point: Temperature scaling
  - τ_s ≈ 0.1: Student sees sharp, definitive labels
  - τ_t ≈ 0.04: Teacher outputs soft labels
  - τ_t < τ_s: Creates asymmetry in distillation

Centering and Sharpening:
==========================

Additional techniques to stabilize training:

1. **Centering**: Subtract mean from predictions
   - Prevents model collapse to uniform distribution
   - Encourages diversified predictions
   - Applied to teacher predictions during loss

2. **Sharpening**: Apply temperature to control distribution
   - Temperature scaling as discussed above
   - Lower temperature: sharper predictions
   - Prevents soft, uncertain predictions

Implementation:
  1. Compute teacher output: z_t
  2. Compute probabilities: p_t = softmax(z_t / τ_t)
  3. Center: p_t_centered = p_t - p_t.mean()
  4. Use centered predictions for loss computation

Why Centering?
  - Without centering: Model can collapse (all predictions uniform)
  - Centering forces diversity in predictions
  - Acts as implicit regularization

ViT Architecture Details:
========================

Typical configuration:
  - Backbone: ViT-Small (21M params) or ViT-Base (86M params)
  - Patch size: 16×16 (creates 196 patches for 224×224 image)
  - Hidden dim: 384 (ViT-S) or 768 (ViT-B)
  - Number of heads: 6 (ViT-S) or 12 (ViT-B)
  - MLP ratio: 4 (hidden dim in feedforward is 4× embedding dim)

Head architecture:
  - Input: [CLS] token embedding [hidden_dim]
  - MLP: Linear(hidden_dim, hidden_dim)
  - Output: Linear(hidden_dim, num_classes)
  - No batch norm in head (unlike SimCLR/BYOL)

Normalization:
  - Layer norm used instead of batch norm
  - ViTs use layer norm throughout
  - Different from CNN-based methods (batch norm)

Training Procedure:
====================

1. Load image and create augmented crops
   - 2 global crops (224×224)
   - 8 local crops (96×96)
   - Total 10 crops per image

2. Pass through networks
   - Student processes all 10 crops
   - Teacher processes only 2 global crops
   - Both produce predictions (after softmax)

3. Compute loss
   - For each student crop: compute cross-entropy with all teacher outputs
   - Cross-entropy between softmax predictions (not logits)
   - Symmetric loss: teacher also sees student outputs (optional)

4. Backprop
   - Only through student network
   - Teacher gets no direct gradients

5. Update teacher
   - After gradient step, update teacher via momentum
   - θ_t ← m·θ_t + (1-m)·θ_s where m ≈ 0.999

Advantages of DINO:
===================
  + Works well with Vision Transformers
  + Natural object discovery (emergent property)
  + Simple cross-entropy loss
  + Multi-crop strategy prevents collapse
  + Scales to large models
  + Linear evaluation competitive with supervised
  + Excellent for transfer learning

Disadvantages of DINO:
=====================
  - More complex augmentation strategy (multi-crop)
  - Requires ViT architecture (not well-studied on CNNs)
  - Computational cost (10 crops per image)
  - Slower training than some alternatives

Downstream Evaluation:
======================

1. **Linear Probing**:
   - Freeze ViT backbone
   - Train only linear classifier on [CLS] token
   - Competitive with supervised ImageNet accuracy

2. **Object Discovery**:
   - Use attention maps as saliency
   - Can segment objects without bounding boxes
   - Qualitatively impressive results

3. **Transfer Learning**:
   - Fine-tune entire model on downstream task
   - Better performance than supervised pretraining
   - Works well with small downstream datasets

Comparison with Other Methods:
==============================

                    DINO        SimCLR      MoCo         BYOL
Paradigm:          Distillation Contrastive Contrastive  Prediction
Requires ViT:       Preferred   CNNs better  Either       Either
#Crops:            10          2            2            2
Negatives:         None        In-batch     Memory bank  None
Loss:              CrossEnt    InfoNCE      Contrastive  MSE
Emergent props:    Yes         No           No           No
Accuracy (IN):     80.1%       69.3%        71.3%        73%

DINO particularly strong on Vision Transformers.

Advanced Topics:
================

1. **Attention Map Analysis**:
   - DINO produces interpretable attention maps
   - Can see which tokens model attends to
   - Often highlights semantic object boundaries
   - Useful for visualization and debugging

2. **Feature Structure**:
   - DINO features show strong semantic clustering
   - Can use k-NN for retrieval without fine-tuning
   - Features work well for various downstream tasks

3. **Scaling Laws**:
   - Larger ViTs benefit more from DINO
   - ViT-B significantly better than ViT-S
   - Scaling to even larger models shows promise

4. **Multi-GPU Training**:
   - DINO works well with distributed training
   - Synchronized operations for batch norm replacement
   - Each GPU processes subset of crops

Practical Implementation Notes:
==============================

1. **Temperature Adjustment**:
   - τ_s = 0.1 (student, sharp)
   - τ_t = 0.04 (teacher, soft)
   - Can decay τ_t during training (warmup)

2. **Momentum Schedule**:
   - m_start = 0.996
   - m_end = 1.0
   - Gradually increase m during training (cosine schedule)
   - Higher momentum later in training for stability

3. **Warmup**:
   - Gradual increase of learning rate first 10 epochs
   - Prevents early instability
   - Important for distributed training

4. **Augmentation Pipeline**:
   - Global crops: Standard augmentation
   - Local crops: Random position, no color jitter
   - Gaussian blur important for local crops

5. **Batch Size**:
   - Need sufficient batch size (512-1024)
   - Multi-GPU training essential for practical use
   - Can use gradient accumulation if needed

Related and Subsequent Work:
===========================

DINO-v2 (2023):
  - More stable training procedure
  - Better hyperparameter choices
  - Improved downstream performance
  - More robust to scaling

DINOx:
  - Distillation across models
  - Different teacher and student architectures
  - Explores knowledge transfer

These build on DINO's foundations but with refinements.

Mathematical Formulation:
========================

The DINO loss combines several components:

1. Student cross-entropy:
   L_s = -Σ_i log[p_s^i] where p_s^i = softmax(z_s / τ_s)

2. Teacher distribution:
   p_t^j = softmax((z_t^j - c) / τ_t + sharpen_term)

3. Centering update:
   c ← λ·c + (1-λ)·z_t.mean()

4. Full loss (symmetric):
   L = L(student, teacher) + L(teacher, student)

5. Overall training objective:
   minimize L + regularization terms

The centering mechanism prevents mode collapse:
  - Without it: model predicts uniform distribution
  - Centering pushes model to use full label space
  - Acts as entropy regularization

Important Implementation Pitfalls:
==================================

1. **Forgetting Centering**:
   - Easy to implement loss without centering
   - Results in poor convergence or collapse
   - Centering is critical

2. **Wrong Temperature Values**:
   - Using τ_s > τ_t: Reversed, won't work well
   - Using values too extreme: Numerical instability
   - Empirical values from paper important

3. **Crop Implementation**:
   - Local crops must be much smaller (96 vs 224)
   - All crops must use same normalization
   - Important for consistency

4. **Momentum Schedule**:
   - Constant momentum may lead to instability
   - Gradually increasing m helps later in training
   - Cosine schedule for m is recommended

5. **Layer Norm vs Batch Norm**:
   - Must use layer norm (part of ViT)
   - Batch norm doesn't work well with layer norm
   - Consistent with ViT architecture
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
import math
from python.nn_core import Module, Parameter


class MultiCropAugmentation:
    """
    DINO's multi-crop augmentation strategy.

    Creates:
    - 2 global crops (224×224)
    - 8 local crops (96×96)
    Each image processes independently through different augmentation.

    Usage:
        aug = MultiCropAugmentation(
            global_size=224,
            local_size=96,
            n_local_crops=8
        )
        crops = aug(image)  # Returns list of 10 augmented crops
    """

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        n_global_crops: int = 2,
        n_local_crops: int = 8
    ):
        """
        Args:
            global_size: Size of global crops (typically 224)
            local_size: Size of local crops (typically 96)
            n_global_crops: Number of global crops (typically 2)
            n_local_crops: Number of local crops (typically 8)
        """
        raise NotImplementedError(
            "Implement multi-crop augmentation:\n"
            "1. Create augmentation for global crops (full image strategy)\n"
            "2. Create augmentation for local crops (small region strategy)\n"
            "3. Both should use: random crop, flip, color jitter, blur\n"
            "4. Local crops: smaller region, no color distortion (optional)\n"
            "5. Return list of n_global + n_local augmented images"
        )

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create multi-crop augmentation of image.

        Args:
            image: Input image tensor [3, H, W]

        Returns:
            List of 10 augmented crops (2 global + 8 local)
        """
        raise NotImplementedError()


class DINOHead(Module):
    """
    DINO projection head applied to ViT features.

    Maps from ViT output embedding to prediction logits.

    Architecture:
      [CLS] token ∈ ℝ^hidden_dim
           → [Linear] → [LayerNorm] → [ReLU]
           → [Linear] → [Softmax] → predictions

    Note: Uses layer norm (not batch norm) consistent with ViT

    Args:
        in_dim: Dimension of ViT output (e.g., 384 for ViT-S)
        out_dim: Number of classes in projection space (typically 65536)
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 65536,
        hidden_dim: int = 384
    ):
        """
        Args:
            in_dim: Input dimension (ViT embedding dimension)
            out_dim: Output dimension (projection classes)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        raise NotImplementedError(
            "Implement DINO head:\n"
            "1. Linear: in_dim → hidden_dim\n"
            "2. LayerNorm (not BatchNorm!)\n"
            "3. ReLU\n"
            "4. Linear: hidden_dim → out_dim\n"
            "Note: Use nn.LayerNorm, not nn.BatchNorm1d"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Get unnormalized logits."""
        raise NotImplementedError()


class DINOStudent(Module):
    """
    Student network for DINO.

    Processes all crops (global + local).

    Architecture:
      Crop → [ViT Backbone] → [DINO Head] → logits [batch_size, out_dim]
    """

    def __init__(
        self,
        backbone: Module,
        head: DINOHead
    ):
        """
        Args:
            backbone: ViT backbone (pre-loaded or created)
            head: DINO head for projection
        """
        super().__init__()
        raise NotImplementedError(
            "Implement student network:\n"
            "1. Store backbone\n"
            "2. Store head"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Get logits for input crops.

        Args:
            x: Input crops [batch_size, 3, H, W]

        Returns:
            Unnormalized logits [batch_size, out_dim]
        """
        raise NotImplementedError(
            "Implement forward:\n"
            "1. x → backbone to get embeddings\n"
            "2. embeddings → head to get logits\n"
            "3. Return logits (not softmax - softmax done in loss)"
        )


class DINOTeacher(Module):
    """
    Teacher network for DINO (momentum updated student).

    Processes only global crops.

    Similar to student but updated via momentum instead of gradient.
    """

    def __init__(
        self,
        backbone: Module,
        head: DINOHead,
        momentum: float = 0.999
    ):
        """
        Args:
            backbone: ViT backbone
            head: DINO head
            momentum: EMA momentum coefficient
        """
        super().__init__()
        raise NotImplementedError(
            "Implement teacher network:\n"
            "1. Store backbone\n"
            "2. Store head\n"
            "3. Store momentum value"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Get logits (with no_grad context)."""
        raise NotImplementedError()

    def update_momentum(self, student: DINOStudent):
        """Update teacher via EMA."""
        raise NotImplementedError(
            "Implement momentum update:\n"
            "For each (teacher_param, student_param):\n"
            "  teacher_param = m * teacher_param + (1-m) * student_param"
        )


class CenteringBuffer(Module):
    """
    Tracks running center (mean) of teacher predictions.

    Used to prevent mode collapse in DINO.

    The center is subtracted from teacher output before softmax,
    encouraging the model to use full label space.

    Implementation:
      c_new = λ·c_old + (1-λ)·batch_mean

    Typical λ ≈ 0.9 (aggressive centering)
    """

    def __init__(
        self,
        num_prototypes: int,
        momentum: float = 0.9,
        device: str = 'cuda'
    ):
        """
        Args:
            num_prototypes: Dimension of center vector (out_dim)
            momentum: Update momentum (default 0.9)
            device: Device to store center on
        """
        super().__init__()
        self.center = np.zeros(num_prototypes, dtype=np.float32)
        self.momentum = momentum
        self.num_updates = 0

    def update(self, z: np.ndarray):
        """
        Update center buffer with batch.

        Args:
            z: Batch of teacher outputs [batch_size, num_prototypes]
        """
        raise NotImplementedError(
            "Update center via momentum:\n"
            "1. Compute batch mean: batch_mean = z.mean(dim=0)\n"
            "2. Update center: center = m*center + (1-m)*batch_mean\n"
            "3. Increment num_updates"
        )

    def get_center(self) -> np.ndarray:
        """Get current center vector."""
        return self.center.copy()


class DINOLoss(Module):
    """
    DINO loss: Cross-entropy between student and teacher predictions.

    Formula:
      L = -Σ_i p_teacher[i] * log(p_student[i])

    Where:
      - p_teacher = softmax((z_teacher - center) / τ_teacher)
      - p_student = softmax(z_student / τ_student)

    Args:
        num_prototypes: Output dimension (typically 65536)
        temperature_student: Temperature for student softmax (default 0.1)
        temperature_teacher: Temperature for teacher softmax (default 0.04)
    """

    def __init__(
        self,
        num_prototypes: int = 65536,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.04
    ):
        """
        Args:
            num_prototypes: Dimensionality of projection space
            temperature_student: Student softmax temperature (sharp)
            temperature_teacher: Teacher softmax temperature (soft)
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau_student = temperature_student
        self.tau_teacher = temperature_teacher
        self.centering_buffer = None

    def set_centering_buffer(self, buffer: CenteringBuffer):
        """Set the centering buffer."""
        self.centering_buffer = buffer

    def forward(
        self,
        student_output: np.ndarray,
        teacher_output: np.ndarray
    ) -> float:
        """
        Compute DINO loss.

        Args:
            student_output: Student logits [total_crops * batch_size, out_dim]
            teacher_output: Teacher logits [n_global_crops * batch_size, out_dim]

        Returns:
            Scalar loss value

        Implementation:
        1. Get center from centering_buffer
        2. Compute teacher probabilities:
           p_t = softmax((teacher_output - center) / tau_teacher)
        3. Compute student probabilities:
           p_s = softmax(student_output / tau_student)
        4. Compute cross-entropy:
           L = -sum(p_t * log(p_s))
        5. Return mean over batch

        Important:
        - Teacher uses centered output
        - Student does not use centering
        - Temperature scaling controls sharpness
        """
        raise NotImplementedError(
            "Implement DINO loss:\n"
            "1. Get center: c = self.centering_buffer.get_center()\n"
            "2. Teacher probs: p_t = softmax((teacher_output - c) / tau_t)\n"
            "3. Student probs: p_s = softmax(student_output / tau_s)\n"
            "4. Cross-entropy: loss = -(p_t * log(p_s)).sum(dim=1).mean()\n"
            "5. Return loss\n"
            "Hint: Use F.cross_entropy with target as probabilities"
        )


class DINOModel(Module):
    """
    Complete DINO model with student and teacher networks.

    Usage:
        model = DINOModel(vit_backbone)
        crops = aug(image)  # Get 10 crops
        student_outs, teacher_outs = model(crops)
        loss = loss_fn(student_outs, teacher_outs)
    """

    def __init__(
        self,
        backbone: Module,
        out_dim: int = 65536,
        hidden_dim: int = 384,
        momentum_teacher: float = 0.999,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.04
    ):
        """
        Args:
            backbone: ViT backbone network
            out_dim: Projection output dimension
            hidden_dim: Head hidden dimension
            momentum_teacher: Momentum for teacher update
            temperature_student: Student temperature
            temperature_teacher: Teacher temperature
        """
        super().__init__()
        raise NotImplementedError(
            "Implement DINO model:\n"
            "1. Get backbone embedding dimension\n"
            "2. Create student network with head\n"
            "3. Create teacher network (copy of student)\n"
            "4. Create centering buffer\n"
            "5. Create loss function with centering buffer\n"
            "6. Store momentum and temperature values"
        )

    def forward(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for all crops.

        Args:
            crops: List of 10 crop tensors [batch_size, 3, H, W]

        Returns:
            student_output: All crop outputs concatenated
            teacher_output: Only global crop outputs
        """
        raise NotImplementedError(
            "Implement forward:\n"
            "1. Process all crops through student: student_outs[]\n"
            "2. Process only first 2 crops through teacher: teacher_outs[]\n"
            "3. Concatenate: student_out = cat(student_outs, dim=0)\n"
            "4. Concatenate: teacher_out = cat(teacher_outs, dim=0)\n"
            "5. Return student_out, teacher_out"
        )

    def update_teacher_momentum(self):
        """Update teacher network via momentum."""
        raise NotImplementedError(
            "Call teacher.update_momentum(student)"
        )

    def update_center(self, teacher_output: np.ndarray):
        """Update centering buffer with batch."""
        raise NotImplementedError(
            "Call centering_buffer.update(teacher_output)"
        )


class DINOTrainer:
    """
    Trainer for DINO self-supervised learning.

    Handles:
    - Multi-crop augmentation
    - Student and teacher forward passes
    - Loss computation with centering
    - Momentum updates
    - Momentum scheduling (linear warmup then increase)

    Usage:
        model = DINOModel(backbone)
        trainer = DINOTrainer(model, train_loader, device='cuda')
        for epoch in range(300):
            train_loss = trainer.train_epoch()
            trainer.update_momentum_schedule(epoch)
    """

    def __init__(
        self,
        model: DINOModel,
        optimizer,
        train_loader,
        augmentation: MultiCropAugmentation,
        loss_fn: DINOLoss,
        device: str = 'cpu',
        n_epochs: int = 300,
        momentum_schedule: Optional[Callable] = None
    ):
        """
        Args:
            model: DINOModel instance
            optimizer: Optimizer (AdamW recommended)
            train_loader: Training data loader
            augmentation: MultiCropAugmentation instance
            loss_fn: DINOLoss instance
            device: 'cpu' (no GPU support in custom Module system)
            n_epochs: Total number of epochs (for momentum schedule)
            momentum_schedule: Optional function(epoch) → momentum value
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.augmentation = augmentation
        self.loss_fn = loss_fn
        self.device = device
        self.n_epochs = n_epochs
        self.momentum_schedule = momentum_schedule or self.default_momentum_schedule
        self.current_epoch = 0

    def default_momentum_schedule(self, epoch: int) -> float:
        """
        Default momentum schedule: linear warmup then increase.

        Formula:
          First 10 epochs: linear from 0.996 to 1.0
          Remaining: constant at high value (0.9999 or higher)

        Args:
            epoch: Current epoch number

        Returns:
            Momentum value for this epoch
        """
        raise NotImplementedError(
            "Implement momentum schedule:\n"
            "1. Warmup (epochs 0-9): linear from 0.996 to 1.0\n"
            "2. Rest: constant high value (0.9999)\n"
            "Formula: m = m_start + (m_end - m_start) * (epoch / warmup_epochs)\n"
            "Hint: Use cosine schedule or linear schedule"
        )

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Training Loop:
        1. Set model to training mode
        2. For each batch of images:
           a. Create multi-crop augmentation (10 crops)
           b. Forward through student with all crops
           c. Forward through teacher with global crops only
           d. Update centering buffer with teacher output
           e. Compute DINO loss
           f. Backward and optimizer step
           g. Update teacher via momentum
           h. Track running loss
        3. Return average loss

        Important Notes:
        - Teacher forward with no_grad
        - Update centering BEFORE computing loss (or AFTER)
        - Update teacher AFTER backward step
        - Multiple crops per image increases gradient steps
        """
        raise NotImplementedError(
            "Implement DINO training loop:\n"
            "1. self.model.train()\n"
            "2. For each batch:\n"
            "   a. Create crops: crops = aug(images)  # List of 10 per image\n"
            "   b. Concatenate batch: [10*B, 3, H, W]\n"
            "   c. Student forward: student_out\n"
            "   d. Teacher forward: teacher_out (no_grad)\n"
            "   e. Update center: model.update_center(teacher_out)\n"
            "   f. Loss: loss = loss_fn(student_out, teacher_out)\n"
            "   g. Backward and step\n"
            "   h. Update teacher: model.update_teacher_momentum()\n"
            "   i. Track loss\n"
            "3. Return average loss"
        )

    def update_momentum_schedule(self, epoch: int):
        """Update momentum based on schedule."""
        raise NotImplementedError(
            "1. Get momentum from schedule: m = self.momentum_schedule(epoch)\n"
            "2. Update model: self.model.momentum_teacher = m"
        )


# ============================================================================
# Key Insights about DINO
# ============================================================================

"""
Why DINO Works So Well with Vision Transformers:

1. **Emergent Object Discovery**:
   - ViTs naturally learn to identify objects
   - Attention maps show semantic understanding
   - No explicit bounding boxes or labels needed
   - This is specific to ViT architecture (not seen as clearly in CNNs)

2. **Multi-Crop Strategy**:
   - Global crops: Broad semantic understanding
   - Local crops: Fine-grained details
   - Student must reconcile both views
   - Creates strong learning signal

3. **Centering Mechanism**:
   - Prevents mode collapse without negatives
   - Encourages use of full label space
   - Simple but effective regularization
   - Acts as implicit entropy constraint

4. **Distillation Framework**:
   - More intuitive than contrastive learning (for some)
   - Clear: "student learns from teacher"
   - Teacher consistency via momentum
   - Natural extension of knowledge distillation

Comparison with SimCLR on ViT:
  - SimCLR: Works but not optimal for ViT
  - DINO: Specifically designed for ViT
  - DINO advantages: Object discovery, scaling

Key Takeaway:
  Different architectures benefit from different pretraining methods.
  Self-supervised learning is not one-size-fits-all.

Practical Tips:
  1. Use ViT architecture (DINO works best with it)
  2. Adjust temperatures empirically for your setting
  3. Centering is critical - don't skip it
  4. Multi-GPU training essential for speed
  5. Momentum schedule can significantly impact results
"""
