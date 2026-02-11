# ML Implementation Curriculum

A structured guide for implementing machine learning from scratch in NumPy.

## How to Use This Curriculum

1. **Follow phases in order** - Each phase builds on previous ones
2. **Complete all ⭐ items before moving on** - These are foundational
3. **Run tests after each implementation** - Verify with `pytest tests/`
4. **Non-⭐ items can be done later** - Come back after completing core path
5. **Expect ~6 months for full completion** - This is a marathon, not a sprint

---

## Phase 1: Foundations (Weeks 1-2) 🔴 CRITICAL

*Build the infrastructure everything depends on.*

### Week 1: Math & Utilities

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Math Utils | Easy | `utils/math_utils.py` | Softmax, logsumexp, numerical stability |
| ⭐ Tensor Utils | Easy | `utils/tensor_utils.py` | Shape checking, broadcasting helpers |
| ⭐ Seeding | Easy | `utils/seeding.py` | Reproducibility utilities |
| ⭐ Data Utils | Easy | `utils/data_utils.py` | Batching, splitting, normalization |
| ⭐ Metrics | Easy | `utils/metrics.py` | Accuracy, F1, MSE, R² |

### Week 2: Automatic Differentiation

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Computational Graph | Hard | `foundations/computational_graph.py` | Tensor class, basic ops |
| ⭐ Autograd | Hard | `foundations/autograd.py` | Reverse-mode AD, gradient tape |
| ⭐ Gradient Check | Easy | `foundations/gradient_check.py` | Numerical gradient verification |

**📖 Reading:**
- Karpathy's micrograd: https://github.com/karpathy/micrograd
- CS231n Backprop: https://cs231n.github.io/optimization-2/
- Understanding Deep Learning Ch. 2-7: https://udlbook.github.io/udlbook/

**✅ Milestone:** Implement basic Tensor with +, *, @, and backward(). Verify with gradient checking.

---

## Phase 2: Core Layers & Training (Weeks 3-4) 🔴 CRITICAL

*Build neural network building blocks.*

### Week 3: Layers & Activations

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Linear Layer | Easy | `nn_core/layers/linear.py` | y = Wx + b with backward |
| ⭐ ReLU | Easy | `nn_core/activations/relu.py` | ReLU, LeakyReLU, PReLU, ELU |
| ⭐ Sigmoid | Easy | `nn_core/activations/sigmoid.py` | Sigmoid, LogSigmoid |
| ⭐ Tanh | Easy | `nn_core/activations/tanh.py` | Tanh, Hardtanh |
| ⭐ Softmax | Medium | `nn_core/activations/softmax.py` | Softmax, LogSoftmax |
| ⭐ GELU | Easy | `nn_core/activations/gelu.py` | GELU for transformers |

### Week 4: Losses & Optimization

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ MSE Loss | Easy | `optimization/losses/mse.py` | Regression loss |
| ⭐ Cross-Entropy | Medium | `optimization/losses/cross_entropy.py` | Classification loss |
| ⭐ SGD | Easy | `optimization/optimizers/sgd.py` | SGD with momentum |
| ⭐ Adam | Medium | `optimization/optimizers/adam.py` | Adam, AdamW |
| ⭐ MLP | Easy | `architectures/mlp.py` | Multi-layer perceptron |

**📖 Reading:**
- Adam paper: https://arxiv.org/abs/1412.6980
- CS231n Neural Networks: https://cs231n.github.io/neural-networks-1/

**✅ Milestone:** Train MLP on MNIST from scratch. Achieve >95% accuracy.

---

## Phase 3: Optimization Deep Dive (Week 5) 🟠

*Master training dynamics.*

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Momentum | Easy | `optimization/optimizers/momentum.py` | Nesterov momentum |
| ⭐ AdamW | Medium | `optimization/optimizers/adam.py` | Decoupled weight decay |
| Schedulers | Easy | `optimization/schedulers/*.py` | Cosine, warmup, step |
| ⭐ BatchNorm | Medium | `nn_core/normalization/batchnorm.py` | Batch normalization |
| ⭐ LayerNorm | Easy | `nn_core/normalization/layernorm.py` | Layer normalization |
| ⭐ Dropout | Easy | `nn_core/regularization/dropout.py` | Regularization |
| Gradient Clipping | Easy | `optimization/gradient_utils/clipping.py` | Prevent exploding gradients |

**✅ Milestone:** Compare SGD vs Adam convergence. Visualize loss landscapes.

---

## Phase 4: Classical ML (Weeks 6-7) 🟠

*Build intuition with simpler algorithms.*

### Week 6: Supervised Learning

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Linear Regression | Easy | `classical_ml/regression/linear.py` | OLS, gradient descent |
| ⭐ Ridge Regression | Easy | `classical_ml/regression/ridge.py` | L2 regularization |
| ⭐ Logistic Regression | Easy | `classical_ml/classification/logistic.py` | Binary/multinomial |
| ⭐ SVM | Medium | `classical_ml/classification/svm.py` | Linear, kernel SVM |
| ⭐ Decision Tree | Medium | `classical_ml/classification/decision_tree.py` | CART algorithm |
| ⭐ Random Forest | Easy | `classical_ml/ensemble/random_forest.py` | Bagging of trees |
| ⭐ KNN | Easy | `classical_ml/classification/knn.py` | K-nearest neighbors |

### Week 7: Unsupervised Learning

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ K-Means | Easy | `classical_ml/clustering/kmeans.py` | K-Means, K-Means++ |
| ⭐ GMM | Medium | `classical_ml/clustering/gmm.py` | EM algorithm |
| ⭐ PCA | Easy | `classical_ml/reduction/pca.py` | Principal components |
| ⭐ t-SNE | Hard | `classical_ml/reduction/tsne.py` | Visualization |
| ⭐ Naive Bayes | Easy | `classical_ml/classification/naive_bayes.py` | Gaussian, multinomial |

**📖 Reading:**
- PRML Bishop Ch. 1-4, 9
- Stanford CS229 Notes: https://cs229.stanford.edu/

**✅ Milestone:** Implement EM for GMM. Visualize clustering on 2D data.

---

## Phase 5: Convolutional Networks (Weeks 8-10) 🔴 CRITICAL

*Spatial processing and image understanding.*

### Week 8-9: Conv Operations

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Conv2D | Hard | `nn_core/conv/conv2d.py` | Forward and backward |
| ⭐ Transposed Conv | Medium | `nn_core/conv/transposed_conv.py` | Upsampling |
| ⭐ MaxPool | Easy | `nn_core/pooling/maxpool.py` | Max pooling |
| ⭐ AvgPool | Easy | `nn_core/pooling/avgpool.py` | Average pooling |
| ⭐ LeNet | Easy | `architectures/lenet.py` | Classic CNN |

### Week 10: Modern Architectures

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ VGG | Easy | `architectures/vgg.py` | Deeper is better |
| ⭐ ResNet | Medium | `architectures/resnet.py` | Skip connections |
| ⭐ U-Net | Medium | `architectures/unet.py` | Encoder-decoder |

**📖 Reading:**
- CS231n CNNs: https://cs231n.github.io/convolutional-networks/
- ResNet paper: https://arxiv.org/abs/1512.03385

**✅ Milestone:** Implement ResNet-18. Train on CIFAR-10, achieve >85% accuracy.

---

## Phase 6: Sequence Models & Attention (Weeks 11-14) 🔴 CRITICAL

*The path to Transformers.*

### Week 11-12: Recurrent Networks

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ RNN Cell | Medium | `nn_core/recurrent/rnn_cell.py` | Basic recurrence |
| ⭐ LSTM Cell | Medium | `nn_core/recurrent/lstm_cell.py` | Long short-term memory |
| ⭐ GRU Cell | Medium | `nn_core/recurrent/gru_cell.py` | Gated recurrent unit |
| ⭐ Seq2Seq | Medium | `sequence/recurrent/seq2seq.py` | Encoder-decoder |

### Week 13-14: Transformers

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Scaled Dot-Product | Easy | `nn_core/attention/scaled_dot_product.py` | Attention mechanism |
| ⭐ Multi-Head | Medium | `nn_core/attention/multihead.py` | Parallel attention |
| ⭐ Positional Encoding | Easy | `nn_core/positional/sinusoidal.py` | Position info |
| ⭐ RoPE | Medium | `nn_core/positional/rope.py` | Rotary embeddings |
| ⭐ Encoder | Medium | `sequence/transformers/encoder.py` | BERT-style |
| ⭐ Decoder | Medium | `sequence/transformers/decoder.py` | GPT-style |
| ⭐ GPT | Hard | `sequence/transformers/gpt.py` | Decoder-only |
| ⭐ ViT | Medium | `sequence/transformers/vit.py` | Vision Transformer |

**📖 Reading:**
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- RoPE paper: https://arxiv.org/abs/2104.09864

**✅ Milestone:** Implement small GPT. Train on tiny Shakespeare, generate text.

---

## Phase 7: Generative Models - VAEs (Week 15) 🟠

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Vanilla AE | Easy | `generative/autoencoders/vanilla_ae.py` | Autoencoder |
| ⭐ VAE | Medium | `generative/autoencoders/vae.py` | Variational AE |
| ⭐ CVAE | Medium | `generative/autoencoders/cvae.py` | Conditional VAE |
| ⭐ VQ-VAE | Hard | `generative/autoencoders/vqvae.py` | Discrete latents |

**📖 Reading:**
- VAE Tutorial: https://arxiv.org/abs/1606.05908
- VQ-VAE paper: https://arxiv.org/abs/1711.00937

**✅ Milestone:** Train VAE on MNIST. Visualize latent space interpolation.

---

## Phase 8: Generative Models - GANs (Weeks 16-17) 🟠

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Vanilla GAN | Medium | `generative/gans/vanilla_gan.py` | Basic GAN |
| ⭐ DCGAN | Medium | `generative/gans/dcgan.py` | Convolutional GAN |
| ⭐ WGAN-GP | Medium | `generative/gans/wgan_gp.py` | Wasserstein + gradient penalty |
| ⭐ Conditional GAN | Medium | `generative/gans/cgan.py` | Class-conditional |

**✅ Milestone:** Train DCGAN on CIFAR-10. Generate plausible images.

---

## Phase 9: Diffusion Models (Weeks 18-20) 🟠

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Noise Schedule | Easy | `generative/diffusion/noise_schedule.py` | Beta schedules |
| ⭐ DDPM | Hard | `generative/diffusion/ddpm.py` | Denoising diffusion |
| ⭐ DDIM | Medium | `generative/diffusion/ddim.py` | Faster sampling |
| ⭐ CFG | Medium | `generative/diffusion/classifier_free_guidance.py` | Guidance |

**📖 Reading:**
- Diffusion Models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- DDPM paper: https://arxiv.org/abs/2006.11239

**✅ Milestone:** Train DDPM on MNIST. Generate samples with DDIM.

---

## Phase 10: Reinforcement Learning (Weeks 21-28) 🔴 FOR ROBOTICS

*Critical for robotics applications.*

### Weeks 21-22: Foundations

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Policies | Easy | `rl/core/policies.py` | Policy representations |
| ⭐ Value Functions | Easy | `rl/core/value_functions.py` | V(s), Q(s,a) |
| ⭐ Replay Buffer | Easy | `rl/core/replay_buffer.py` | Experience replay |
| ⭐ Q-Learning | Easy | `rl/tabular/q_learning.py` | Tabular Q-learning |
| ⭐ Value Iteration | Easy | `rl/tabular/value_iteration.py` | Dynamic programming |

### Week 23: Policy Gradient

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ REINFORCE | Easy | `rl/policy_gradient/reinforce.py` | Basic policy gradient |
| ⭐ A2C | Medium | `rl/policy_gradient/a2c.py` | Advantage actor-critic |
| ⭐ PPO | Medium | `rl/policy_gradient/ppo.py` | Proximal policy optimization |

### Weeks 24-25: Deep RL

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ DQN | Medium | `rl/value_based/dqn.py` | Deep Q-network |
| ⭐ Double DQN | Easy | `rl/value_based/double_dqn.py` | Overestimation fix |
| ⭐ DDPG | Medium | `rl/actor_critic/ddpg.py` | Continuous control |
| ⭐ TD3 | Medium | `rl/actor_critic/td3.py` | Twin delayed DDPG |
| ⭐ SAC | Hard | `rl/actor_critic/sac.py` | Soft actor-critic |

### Weeks 26-28: Robotics RL

| Module | Difficulty | File | Description |
|--------|------------|------|-------------|
| ⭐ Model-Based | Hard | `rl/model_based/mbpo.py` | Sample efficiency |
| ⭐ Behavior Cloning | Easy | `rl/imitation/behavior_cloning.py` | From demonstrations |
| ⭐ GAIL | Hard | `rl/imitation/gail.py` | Adversarial imitation |
| ⭐ CQL | Hard | `rl/offline/cql.py` | Conservative Q-learning |
| ⭐ HER | Medium | `rl/multi_goal/her.py` | Hindsight experience replay |

**📖 Reading:**
- Spinning Up: https://spinningup.openai.com/
- Sutton & Barto: http://incompleteideas.net/book/the-book.html
- SAC paper: https://arxiv.org/abs/1801.01290

**✅ Milestones:**
- Train PPO on CartPole from scratch
- Train SAC on Pendulum
- Implement HER for sparse-reward reaching

---

## Phase 11: Advanced Topics (Weeks 29+) 🟢

*Choose based on interests.*

### Self-Supervised Learning
- SimCLR, MoCo, BYOL
- CLIP concepts
- MAE

### Efficient ML
- Pruning, quantization
- Knowledge distillation

### 3D Vision
- NeRF basics
- Volume rendering

### Graph Neural Networks
- GCN, GAT, GraphSAGE

### Meta-Learning
- MAML, Prototypical Networks

---

## Tips for Success

### General
- Start simple, verify with gradient checking
- Test obsessively with small inputs
- Visualize everything
- Read original papers

### Common Pitfalls
- Numerical instability: use logsumexp
- Broadcasting bugs: print shapes
- Forgetting to zero gradients
- Bad initialization preventing learning

### When Stuck
1. Simplify to smallest failing case
2. Check shapes at every step
3. Compare to PyTorch reference
4. Re-read the paper/textbook

---

## Time Estimates

| Phase | Weeks | Hours/Week | Total Hours |
|-------|-------|------------|-------------|
| Foundations | 2 | 10-15 | 20-30 |
| Core Layers | 2 | 10-15 | 20-30 |
| Optimization | 1 | 10-15 | 10-15 |
| Classical ML | 2 | 10-15 | 20-30 |
| CNNs | 3 | 10-15 | 30-45 |
| Transformers | 4 | 10-15 | 40-60 |
| Generative | 6 | 10-15 | 60-90 |
| RL | 8 | 10-15 | 80-120 |
| Advanced | 4+ | 10-15 | 40+ |
| **Total** | **~32** | | **~320-420** |

---

*Good luck! The journey of implementing ML from scratch is challenging but incredibly rewarding.*
