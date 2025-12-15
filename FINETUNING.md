# Fine-Tuning Guide for SEDD Models

This guide explains how to perform Supervised Fine-Tuning (SFT) on pre-trained SEDD models (small/medium) and discusses potential Reinforcement Learning (RL) methods for discrete diffusion language models.

## Table of Contents

1. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
   - [Quick Start](#quick-start)
   - [Loading Pretrained Models](#loading-pretrained-models)
   - [Dataset Preparation](#dataset-preparation)
   - [Running SFT Training](#running-sft-training)
   - [Configuration Options](#configuration-options)
   - [Recommended SFT Datasets](#recommended-sft-datasets)
2. [Reinforcement Learning Methods](#reinforcement-learning-methods)
   - [Challenges with RL for Discrete Diffusion](#challenges-with-rl-for-discrete-diffusion)
   - [Potential RL Approaches](#potential-rl-approaches)
   - [Implementation Considerations](#implementation-considerations)

---

## Supervised Fine-Tuning (SFT)

### Quick Start

To fine-tune a pretrained SEDD model on a custom dataset (e.g., S1K-1.1), simply run:

```bash
# Fine-tune SEDD-small on S1K-1.1 dataset
python sft.py sft.pretrained_model=louaaron/sedd-small sft.dataset=simplescaling/s1K-1.1

# Fine-tune SEDD-medium on a custom dataset
python sft.py sft.pretrained_model=louaaron/sedd-medium sft.dataset=your-org/your-dataset
```

### Loading Pretrained Models

SEDD provides pretrained models hosted on HuggingFace:
- **SEDD-small**: `louaaron/sedd-small` (~125M parameters)
- **SEDD-medium**: `louaaron/sedd-medium` (~350M parameters)

You can also fine-tune from a locally trained checkpoint:

```bash
# From a local checkpoint
python sft.py sft.pretrained_model=exp_local/2024.01.15/120000
```

The SFT script automatically detects whether to load from HuggingFace or a local path.

### Dataset Preparation

The SFT script supports any HuggingFace dataset with text content. The script will automatically:

1. Load the dataset from HuggingFace Hub
2. Tokenize using GPT-2 tokenizer (same as pretraining)
3. For Q&A datasets: separate prompts from targets and apply loss only to target tokens
4. Pad sequences to `max_length` with proper attention masking
5. Create train/validation splits if not provided

**Supported text field configurations:**
- `sft.text_field`: Single text field (e.g., `text`, `content`)
- `sft.question_field` + `sft.response_field`: Q&A format with explicit fields
- `sft.question_field` + `sft.thinking_field` + `sft.attempt_field`: For datasets with reasoning traces (like S1K)
- Auto-detection: `question` + `answer`, or `question` + `solution`

**Key Features:**
- **Prompt Masking**: Loss is computed only on response tokens (prompt tokens are masked with -100)
- **No Chunking**: Each example is kept intact to preserve question-answer alignment
- **Proper Padding**: Sequences are padded to `max_length` with attention masks
- **EOS Token**: Each sequence ends with an EOS token for proper generation

#### Using S1K-1.1 Dataset

The [S1K-1.1 dataset](https://huggingface.co/datasets/simplescaling/s1K-1.1) contains math reasoning problems with multiple fields:

| Field | Description | Recommended for SFT |
|-------|-------------|---------------------|
| `question` | The problem statement | ✅ Use as input |
| `solution` | Ground truth solution | ✅ Use as target (recommended) |
| `gemini_thinking_trajectory` | Gemini's reasoning trace | ⚠️ Alternative for reasoning-focused SFT |
| `gemini_attempt` | Gemini's final response | ⚠️ Alternative target |
| `deepseek_thinking_trajectory` | DeepSeek's reasoning trace | ⚠️ Alternative for reasoning-focused SFT |
| `deepseek_attempt` | DeepSeek's final response | ⚠️ Alternative target |

**Recommended configurations for S1K-1.1:**

```bash
# Option 1: Use question + solution (recommended for standard SFT)
python sft.py \
    sft.pretrained_model=louaaron/sedd-small \
    sft.dataset=simplescaling/s1K-1.1 \
    sft.question_field=question \
    sft.response_field=solution

# Option 2: Use question + thinking_trajectory + attempt (for reasoning-focused SFT)
# This concatenates thinking_trajectory and attempt as the target with labels
python sft.py \
    sft.pretrained_model=louaaron/sedd-small \
    sft.dataset=simplescaling/s1K-1.1 \
    sft.question_field=question \
    sft.thinking_field=gemini_thinking_trajectory \
    sft.attempt_field=gemini_attempt

# Option 3: Use DeepSeek's reasoning traces
python sft.py \
    sft.pretrained_model=louaaron/sedd-small \
    sft.dataset=simplescaling/s1K-1.1 \
    sft.question_field=question \
    sft.thinking_field=deepseek_thinking_trajectory \
    sft.attempt_field=deepseek_attempt

# Option 4: Auto-detection (uses question + solution automatically)
python sft.py \
    sft.pretrained_model=louaaron/sedd-small \
    sft.dataset=simplescaling/s1K-1.1
```

**Choosing the right fields:**
- **For standard answer generation**: Use `question` + `solution` - trains the model to produce correct answers
- **For reasoning/chain-of-thought**: Use `question` + `thinking_field` + `attempt_field` - trains the model to show its reasoning process followed by the answer
  - The model will learn to generate: `Thinking: <reasoning_trace>\n\nAttempt: <final_answer>`
  - Loss is computed only on the response (thinking + attempt), not on the question prompt
- **For single-field reasoning**: Use `question` + `thinking_trajectory` field as response_field - trains on reasoning trace only
- **For concise responses**: Use `question` + `attempt` field as response_field - trains on final answers only

### Running SFT Training

**Basic SFT command:**

```bash
python sft.py sft.pretrained_model=louaaron/sedd-small sft.dataset=your-dataset
```

**Multi-GPU training:**

```bash
python sft.py ngpus=4 sft.pretrained_model=louaaron/sedd-medium sft.dataset=your-dataset
```

**Resume from checkpoint:**

```bash
python sft.py load_dir=exp_local/sft/your-dataset/2024.01.15/120000
```

**SLURM submission:**

```bash
python sft.py -m sft.pretrained_model=louaaron/sedd-small sft.dataset=your-dataset
```

### Configuration Options

The SFT configuration (`configs/sft.yaml`) provides the following options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sft.pretrained_model` | `louaaron/sedd-small` | Path to pretrained model (HuggingFace or local) |
| `sft.freeze_embeddings` | `False` | Whether to freeze embedding layers |
| `sft.dataset` | `simplescaling/s1K-1.1` | HuggingFace dataset name |
| `sft.text_field` | `text` | Name of the text field (for single-field datasets) |
| `sft.question_field` | `null` | Question field name (for Q&A datasets) |
| `sft.response_field` | `null` | Response/solution field name (for Q&A datasets) |
| `sft.thinking_field` | `null` | Thinking trajectory field name (for S1K-like datasets with reasoning traces) |
| `sft.attempt_field` | `null` | Attempt/answer field name (for S1K-like datasets, used with thinking_field) |
| `sft.max_length` | `null` | Max sequence length (uses model default if null) |
| `optim.lr` | `1e-5` | Learning rate (lower than pretraining) |
| `optim.warmup` | `500` | Warmup steps |
| `training.n_iters` | `50000` | Total training iterations |
| `training.batch_size` | `64` | Batch size |
| `training.accum` | `1` | Gradient accumulation steps |

**Example: Custom configuration**

```bash
python sft.py \
    sft.pretrained_model=louaaron/sedd-medium \
    sft.dataset=simplescaling/s1K-1.1 \
    sft.question_field=question \
    sft.response_field=solution \
    sft.freeze_embeddings=True \
    optim.lr=5e-6 \
    training.batch_size=32 \
    training.n_iters=100000
```

### Recommended SFT Datasets

Here are datasets suitable for fine-tuning SEDD models, organized by task:

#### Reasoning & Math
| Dataset | Description | HuggingFace Path |
|---------|-------------|------------------|
| S1K-1.1 | Curated reasoning examples | `simplescaling/s1K-1.1` |
| GSM8K | Grade school math problems | `gsm8k` |
| MATH | Competition math problems | `hendrycks/competition_math` |
| MetaMathQA | Math reasoning augmentation | `meta-math/MetaMathQA` |

#### Code
| Dataset | Description | HuggingFace Path |
|---------|-------------|------------------|
| CodeAlpaca | Code instruction following | `sahil2801/CodeAlpaca-20k` |
| StarCoder Self-Instruct | Code generation | `bigcode/self-oss-instruct-sc2-exec-filter-50k` |

#### Instruction Following
| Dataset | Description | HuggingFace Path |
|---------|-------------|------------------|
| Alpaca | General instruction following | `tatsu-lab/alpaca` |
| Dolly | Databricks instruction data | `databricks/databricks-dolly-15k` |
| FLAN | Task-oriented instructions | `Muennighoff/flan` |
| OpenAssistant | Conversational data | `OpenAssistant/oasst1` |

#### Domain-Specific
| Dataset | Description | HuggingFace Path |
|---------|-------------|------------------|
| PubMedQA | Medical Q&A | `pubmed_qa` |
| FinQA | Financial reasoning | `dreamerdeo/finqa` |
| SciQ | Science questions | `sciq` |

**Example usage for different datasets:**

```bash
# Math reasoning
python sft.py sft.dataset=gsm8k sft.text_field=question

# Code generation
python sft.py sft.dataset=sahil2801/CodeAlpaca-20k sft.text_field=instruction

# Instruction following
python sft.py sft.dataset=tatsu-lab/alpaca sft.text_field=text
```

---

## Reinforcement Learning Methods

Applying RL methods to discrete diffusion language models like SEDD is an active research area. Here we discuss potential approaches and considerations.

### Challenges with RL for Discrete Diffusion

Unlike autoregressive models, SEDD generates text through an iterative denoising process:

1. **Non-sequential generation**: Tokens are generated in parallel, not left-to-right
2. **Discrete state space**: The score function operates on discrete tokens
3. **Multi-step denoising**: Generation involves multiple diffusion steps
4. **Credit assignment**: Difficult to assign reward credit across diffusion steps

### Potential RL Approaches

#### 1. Reward-Weighted Denoising (RWD)

**Concept**: Weight training samples by their reward scores, similar to reward-weighted regression.

```python
# Pseudocode for reward-weighted training
def reward_weighted_loss(model, batch, reward_fn):
    # Generate samples from current model
    samples = sample_from_model(model, batch.shape)
    
    # Compute rewards
    rewards = reward_fn(samples)
    
    # Weight the standard SEDD loss by rewards
    loss = sedd_loss(model, batch)
    weighted_loss = (rewards * loss).mean()
    
    return weighted_loss
```

#### 2. Diffusion Policy Optimization (DPO for Diffusion)

**Concept**: Adapt Direct Preference Optimization to diffusion models by comparing preferred vs. rejected denoising trajectories.

**Key idea**: Train the model to assign higher likelihood to preferred outputs by modifying the score function.

```python
# Pseudocode for Diffusion DPO
def diffusion_dpo_loss(model, preferred, rejected, beta=0.1):
    # Compute log probabilities under the model
    log_p_preferred = compute_sequence_log_prob(model, preferred)
    log_p_rejected = compute_sequence_log_prob(model, rejected)
    
    # DPO objective
    loss = -log_sigmoid(beta * (log_p_preferred - log_p_rejected))
    
    return loss.mean()
```

#### 3. Score Function Gradient Estimation

**Concept**: Use policy gradient methods by treating the denoising process as a policy.

**Steps**:
1. Sample denoising trajectories
2. Evaluate final samples with reward
3. Estimate gradients using REINFORCE or similar

```python
# Pseudocode for policy gradient
def policy_gradient_step(model, reward_fn, n_samples):
    # Sample trajectories
    trajectories, log_probs = sample_with_log_prob(model, n_samples)
    
    # Get rewards for final samples
    rewards = reward_fn(trajectories[-1])
    
    # Normalize rewards (baseline subtraction)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Policy gradient loss
    loss = -(log_probs * rewards.detach()).mean()
    
    return loss
```

#### 4. Classifier-Free Guidance with Reward

**Concept**: Use reward models to guide the denoising process without retraining.

```python
# Pseudocode for reward-guided sampling
def reward_guided_sample(model, reward_model, guidance_scale=1.0):
    x = sample_initial_noise()
    
    for t in timesteps:
        # Unconditional score
        score_uncond = model.score(x, t)
        
        # Reward gradient (approximate)
        reward_grad = compute_reward_gradient(reward_model, x)
        
        # Guided score
        score = score_uncond + guidance_scale * reward_grad
        
        x = denoise_step(x, score, t)
    
    return x
```

#### 5. Reinforced Fine-Tuning (ReFT)

**Concept**: Alternate between generating samples and fine-tuning on high-reward samples.

```python
# Pseudocode for ReFT
def reft_training(model, reward_fn, n_iterations):
    for i in range(n_iterations):
        # Generate samples
        samples = generate_samples(model, n=1000)
        
        # Score samples
        rewards = reward_fn(samples)
        
        # Select top-k samples
        top_samples = select_top_k(samples, rewards, k=100)
        
        # Fine-tune on top samples (standard SFT)
        sft_step(model, top_samples)
```

### Implementation Considerations

#### Reward Model Selection

For RLHF with SEDD, you'll need a reward model. Options include:

1. **Use existing reward models**: Models trained for autoregressive LLMs can evaluate SEDD outputs
2. **Train custom reward models**: Fine-tune on preference data specific to your task
3. **Use proxy metrics**: Perplexity, BLEU, or task-specific metrics

#### Practical Tips

1. **Start with SFT**: Always fine-tune with SFT before RL
2. **Low learning rates**: Use very low LR for RL (1e-6 or lower)
3. **KL penalty**: Add KL divergence penalty to prevent mode collapse
4. **Gradient clipping**: Essential for stable RL training
5. **Checkpoint frequently**: RL can be unstable

#### Research Directions

Current research on RL for discrete diffusion is limited. Key papers to follow:

- "Diffusion Models for Reinforcement Learning" (Janner et al., 2022)
- "Diffusion Policy" (Chi et al., 2023)
- "Training Diffusion Models with Reinforcement Learning" (Black et al., 2023)
- "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (Lou et al., 2024)

---

## Example Workflows

### Workflow 1: SFT on Reasoning Data

```bash
# Step 1: Fine-tune on S1K-1.1
python sft.py \
    sft.pretrained_model=louaaron/sedd-small \
    sft.dataset=simplescaling/s1K-1.1 \
    training.n_iters=50000

# Step 2: Generate samples for evaluation
python run_sample.py --model_path exp_local/sft/simplescaling/s1K-1.1/.../

# Step 3: Continue fine-tuning on math data
python sft.py \
    sft.pretrained_model=exp_local/sft/simplescaling/s1K-1.1/.../ \
    sft.dataset=gsm8k \
    training.n_iters=25000
```

### Workflow 2: Multi-Stage Fine-Tuning

```bash
# Stage 1: General instruction following
python sft.py \
    sft.pretrained_model=louaaron/sedd-medium \
    sft.dataset=tatsu-lab/alpaca \
    training.n_iters=30000

# Stage 2: Task-specific fine-tuning
python sft.py \
    sft.pretrained_model=exp_local/sft/alpaca/.../ \
    sft.dataset=your-task-dataset \
    training.n_iters=20000
```

---

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `training.batch_size`
- Increase `training.accum` for gradient accumulation
- Use SEDD-small instead of SEDD-medium

**Slow Training**
- Increase `ngpus` for multi-GPU training
- Reduce `training.eval_freq` and `training.snapshot_freq`

**Poor Results**
- Increase `training.n_iters`
- Try different learning rates
- Ensure dataset quality

**Dataset Loading Errors**
- Check `sft.text_field` matches your dataset
- Verify dataset path on HuggingFace

---

## Citation

If you use this fine-tuning code, please cite:

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
