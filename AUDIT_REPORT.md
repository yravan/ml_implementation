# Codebase Audit Report

## Summary

- Total modules: 20
- Total Python files: 265
- Files with content: 171
- Total NotImplementedError stubs: 1756
- Modules with tests: 15/20

## Issues

- [efficient_ml] No test file found
- [experiments] No test file found
- [geometry] No test file found
- [graph] No test file found
- [visualization] No test file found

## Module Details

### architectures

| File | Lines | NotImplemented |
|------|-------|----------------|
| architectures/mlp.py | 143 | 12 |

### bayesian

| File | Lines | NotImplemented |
|------|-------|----------------|
| bayesian/bnn.py | 131 | 13 |

### classical_ml

| File | Lines | NotImplemented |
|------|-------|----------------|
| classical_ml/classification/logistic.py | 227 | 16 |
| classical_ml/classification/naive_bayes.py | 183 | 10 |
| classical_ml/classification/decision_tree.py | 160 | 11 |
| classical_ml/classification/svm.py | 219 | 14 |
| classical_ml/classification/knn.py | 168 | 11 |
| classical_ml/clustering/gmm.py | 322 | 15 |
| classical_ml/clustering/kmeans.py | 245 | 14 |
| classical_ml/clustering/spectral.py | 115 | 13 |
| classical_ml/clustering/dbscan.py | 238 | 12 |
| classical_ml/reduction/lda.py | 301 | 16 |
| classical_ml/reduction/pca.py | 139 | 17 |
| classical_ml/reduction/tsne.py | 125 | 14 |
| classical_ml/regression/gaussian_process.py | 115 | 9 |
| classical_ml/regression/linear.py | 236 | 12 |
| classical_ml/regression/ridge.py | 135 | 9 |
| classical_ml/regression/lasso.py | 110 | 10 |
| classical_ml/regression/polynomial.py | 102 | 11 |

### efficient_ml

| File | Lines | NotImplemented |
|------|-------|----------------|

### envs

| File | Lines | NotImplemented |
|------|-------|----------------|
| envs/gridworld.py | 101 | 9 |

### experiments

| File | Lines | NotImplemented |
|------|-------|----------------|

### foundations

| File | Lines | NotImplemented |
|------|-------|----------------|
| foundations/gradient_check.py | 149 | 11 |
| foundations/computational_graph.py | 183 | 0 |
| foundations/autograd.py | 96 | 11 |

### generative

| File | Lines | NotImplemented |
|------|-------|----------------|
| generative/ebm/contrastive_divergence.py | 171 | 12 |
| generative/ebm/ebm.py | 143 | 20 |
| generative/ebm/langevin.py | 183 | 10 |
| generative/diffusion/ddim.py | 127 | 11 |
| generative/diffusion/noise_schedule.py | 125 | 20 |
| generative/diffusion/classifier_free_guidance.py | 141 | 10 |
| generative/diffusion/latent_diffusion.py | 209 | 20 |
| generative/diffusion/ddpm.py | 156 | 13 |
| generative/diffusion/score_matching.py | 172 | 17 |
| generative/autoencoders/cvae.py | 153 | 14 |
| generative/autoencoders/beta_vae.py | 121 | 13 |
| generative/autoencoders/denoising_ae.py | 73 | 9 |
| generative/autoencoders/vanilla_ae.py | 57 | 7 |
| generative/autoencoders/vqvae.py | 165 | 15 |
| generative/autoencoders/vae.py | 124 | 12 |
| generative/gans/wgan.py | 129 | 11 |
| generative/gans/dcgan.py | 101 | 9 |
| generative/gans/sngan.py | 127 | 14 |
| generative/gans/cgan.py | 127 | 10 |
| generative/gans/wgan_gp.py | 207 | 11 |
| generative/gans/vanilla_gan.py | 99 | 10 |
| generative/gans/pix2pix.py | 102 | 10 |
| generative/gans/cyclegan.py | 135 | 12 |
| generative/flows/glow.py | 217 | 27 |
| generative/flows/realnvp.py | 194 | 18 |
| generative/flows/nice.py | 170 | 17 |
| generative/flows/change_of_variables.py | 90 | 10 |
| generative/flows/flow_matching.py | 236 | 26 |

### geometry

| File | Lines | NotImplemented |
|------|-------|----------------|

### graph

| File | Lines | NotImplemented |
|------|-------|----------------|

### interpretability

| File | Lines | NotImplemented |
|------|-------|----------------|
| interpretability/saliency.py | 122 | 8 |

### meta_learning

| File | Lines | NotImplemented |
|------|-------|----------------|
| meta_learning/maml.py | 195 | 14 |

### nn_core

| File | Lines | NotImplemented |
|------|-------|----------------|
| nn_core/activations/tanh.py | 44 | 11 |
| nn_core/activations/gelu.py | 72 | 8 |
| nn_core/activations/relu.py | 96 | 18 |
| nn_core/activations/sigmoid.py | 62 | 11 |
| nn_core/activations/softmax.py | 78 | 13 |
| nn_core/init/kaiming.py | 94 | 9 |
| nn_core/init/normal.py | 131 | 15 |
| nn_core/init/xavier.py | 104 | 10 |
| nn_core/init/orthogonal.py | 98 | 11 |
| nn_core/attention/grouped_query.py | 90 | 4 |
| nn_core/attention/multihead.py | 62 | 2 |
| nn_core/attention/causal_mask.py | 87 | 6 |
| nn_core/attention/scaled_dot_product.py | 34 | 2 |
| nn_core/attention/cross_attention.py | 86 | 4 |
| nn_core/attention/multi_query.py | 65 | 3 |
| nn_core/pooling/adaptive_pool.py | 101 | 11 |
| nn_core/pooling/global_pool.py | 109 | 12 |
| nn_core/pooling/avgpool.py | 126 | 10 |
| nn_core/pooling/maxpool.py | 109 | 8 |
| nn_core/layers/linear.py | 95 | 9 |
| nn_core/normalization/rmsnorm.py | 69 | 6 |
| nn_core/normalization/spectralnorm.py | 112 | 5 |
| nn_core/normalization/layernorm.py | 73 | 4 |
| nn_core/normalization/batchnorm.py | 122 | 5 |
| nn_core/normalization/groupnorm.py | 83 | 5 |
| nn_core/regularization/dropout2d.py | 161 | 10 |
| nn_core/regularization/label_smoothing.py | 83 | 7 |
| nn_core/regularization/droppath.py | 151 | 8 |
| nn_core/regularization/dropout.py | 120 | 7 |
| nn_core/conv/transposed_conv.py | 120 | 2 |
| nn_core/conv/conv1d.py | 89 | 2 |
| nn_core/conv/conv2d.py | 145 | 2 |
| nn_core/conv/dilated_conv.py | 164 | 3 |
| nn_core/conv/depthwise_separable.py | 169 | 6 |
| nn_core/recurrent/rnn_cell.py | 97 | 4 |
| nn_core/recurrent/gru_cell.py | 97 | 4 |
| nn_core/recurrent/bidirectional.py | 87 | 4 |
| nn_core/recurrent/lstm_cell.py | 125 | 4 |
| nn_core/positional/sinusoidal.py | 46 | 5 |
| nn_core/positional/learned.py | 86 | 6 |
| nn_core/positional/rope.py | 101 | 8 |
| nn_core/positional/alibi.py | 156 | 8 |

### optimization

| File | Lines | NotImplemented |
|------|-------|----------------|
| optimization/losses/cross_entropy.py | 130 | 14 |
| optimization/losses/mse.py | 83 | 8 |
| optimization/optimizers/sgd.py | 91 | 9 |
| optimization/optimizers/adam.py | 121 | 9 |

### rl

| File | Lines | NotImplemented |
|------|-------|----------------|
| rl/hierarchical/hrl.py | 240 | 20 |
| rl/value_based/prioritized_dqn.py | 209 | 15 |
| rl/value_based/dueling_dqn.py | 113 | 7 |
| rl/value_based/dqn.py | 275 | 19 |
| rl/value_based/double_dqn.py | 86 | 6 |
| rl/core/value_functions.py | 222 | 17 |
| rl/core/networks.py | 184 | 15 |
| rl/core/utils.py | 87 | 13 |
| rl/core/replay_buffer.py | 179 | 17 |
| rl/core/advantage.py | 119 | 8 |
| rl/core/policies.py | 102 | 12 |
| rl/bandits/linucb.py | 115 | 12 |
| rl/bandits/thompson_sampling.py | 117 | 17 |
| rl/bandits/neural_bandits.py | 189 | 12 |
| rl/bandits/epsilon_greedy.py | 118 | 15 |
| rl/bandits/ucb.py | 127 | 15 |
| rl/tabular/td_learning.py | 191 | 20 |
| rl/tabular/policy_iteration.py | 123 | 14 |
| rl/tabular/monte_carlo.py | 163 | 19 |
| rl/tabular/policy_evaluation.py | 133 | 10 |
| rl/tabular/value_iteration.py | 143 | 13 |
| rl/policy_gradient/vpg.py | 223 | 19 |
| rl/policy_gradient/npg.py | 116 | 14 |
| rl/policy_gradient/reinforce.py | 196 | 15 |
| rl/policy_gradient/a2c.py | 132 | 18 |
| rl/exploration/intrinsic_motivation.py | 172 | 17 |
| rl/actor_critic/td3.py | 202 | 10 |
| rl/actor_critic/sac.py | 319 | 9 |
| rl/actor_critic/ddpg.py | 253 | 18 |
| rl/multi_goal/goal_conditioned.py | 164 | 14 |
| rl/model_based/cem.py | 166 | 11 |
| rl/model_based/mppi.py | 184 | 12 |
| rl/model_based/world_model.py | 222 | 13 |
| rl/model_based/mpc.py | 202 | 13 |
| rl/imitation/behavior_cloning.py | 215 | 15 |
| rl/imitation/dagger.py | 184 | 16 |
| rl/imitation/gail.py | 250 | 17 |
| rl/offline/iql.py | 94 | 12 |
| rl/offline/bcq.py | 118 | 17 |
| rl/offline/cql.py | 100 | 11 |

### robustness

| File | Lines | NotImplemented |
|------|-------|----------------|
| robustness/attacks/adversarial.py | 144 | 8 |

### self_supervised

| File | Lines | NotImplemented |
|------|-------|----------------|
| self_supervised/masked/mae.py | 218 | 16 |
| self_supervised/masked/masked_lm.py | 198 | 11 |
| self_supervised/multimodal/clip.py | 194 | 13 |
| self_supervised/contrastive/byol.py | 156 | 18 |
| self_supervised/contrastive/dino.py | 182 | 18 |
| self_supervised/contrastive/moco.py | 148 | 11 |
| self_supervised/contrastive/simclr.py | 97 | 11 |
| self_supervised/contrastive/infonce.py | 68 | 6 |

### sequence

| File | Lines | NotImplemented |
|------|-------|----------------|
| sequence/temporal/tcn.py | 76 | 0 |
| sequence/ssm/s4.py | 68 | 0 |
| sequence/ssm/mamba.py | 80 | 0 |
| sequence/ssm/linear_ssm.py | 58 | 0 |
| sequence/recurrent/vanilla_rnn.py | 52 | 0 |
| sequence/recurrent/lstm.py | 62 | 0 |
| sequence/recurrent/gru.py | 59 | 0 |
| sequence/recurrent/seq2seq.py | 85 | 0 |
| sequence/transformers/decoder.py | 300 | 10 |
| sequence/transformers/encoder_decoder.py | 303 | 8 |
| sequence/transformers/encoder.py | 209 | 11 |
| sequence/transformers/vit.py | 158 | 6 |
| sequence/transformers/bert.py | 264 | 16 |
| sequence/transformers/gpt.py | 270 | 9 |
| sequence/efficient/longformer.py | 61 | 0 |
| sequence/efficient/linear_attention.py | 57 | 0 |
| sequence/moe/moe_layer.py | 77 | 0 |
| sequence/moe/top_k_gating.py | 61 | 0 |

### utils

| File | Lines | NotImplemented |
|------|-------|----------------|
| utils/seeding.py | 25 | 0 |
| utils/metrics.py | 147 | 0 |
| utils/math_utils.py | 28 | 0 |
| utils/tensor_utils.py | 85 | 5 |
| utils/data_utils.py | 122 | 0 |

### visualization

| File | Lines | NotImplemented |
|------|-------|----------------|

