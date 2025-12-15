"""Supervised Fine-Tuning (SFT) for SEDD models"""

import datetime
import os
import os.path
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler


torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def get_sft_dataset(dataset_name, text_field, tokenizer, max_length, cache_dir=None, num_proc=8, question_field=None, response_field=None):
    """Load and prepare an SFT dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        text_field: Field containing text (for single-field datasets)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        cache_dir: Cache directory for datasets
        num_proc: Number of processes for preprocessing
        question_field: Field containing questions (for Q&A datasets)
        response_field: Field containing responses/solutions (for Q&A datasets)
    """
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Get train and validation splits
    if "train" in dataset:
        train_data = dataset["train"]
    else:
        # If no train split, use the main dataset
        train_data = dataset
    
    if "validation" in dataset:
        valid_data = dataset["validation"]
    elif "test" in dataset:
        valid_data = dataset["test"]
    else:
        # Create a small validation set from train
        splits = train_data.train_test_split(test_size=0.05, seed=42)
        train_data = splits["train"]
        valid_data = splits["test"]
    
    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    
    def preprocess_and_tokenize(example):
        # Handle different text field configurations
        # Priority 1: Explicit question_field + response_field (for Q&A datasets like S1K-1.1)
        if question_field is not None and response_field is not None:
            if question_field in example and response_field in example:
                text = [f"Question: {q}\n\nAnswer: {a}" for q, a in zip(example[question_field], example[response_field])]
            else:
                raise ValueError(f"Specified fields not found. question_field='{question_field}', response_field='{response_field}'. Available: {list(example.keys())}")
        # Priority 2: Explicit text_field
        elif text_field in example:
            text = example[text_field]
        # Priority 3: Common field names
        elif "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "question" in example and "answer" in example:
            # For Q&A datasets with standard naming
            text = [f"Question: {q}\n\nAnswer: {a}" for q, a in zip(example["question"], example["answer"])]
        elif "question" in example and "solution" in example:
            # For datasets like S1K-1.1 with question/solution format
            text = [f"Question: {q}\n\nSolution: {s}" for q, s in zip(example["question"], example["solution"])]
        else:
            # Try to find any text-like field
            for key in example.keys():
                if isinstance(example[key], list) and len(example[key]) > 0 and isinstance(example[key][0], str):
                    text = example[key]
                    break
            else:
                raise ValueError(f"Could not find text field in dataset. Available fields: {list(example.keys())}")
        
        tokens = tokenizer(text, return_attention_mask=False, truncation=True, max_length=max_length-1)
        # Add EOS token
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    train_tokenized = train_data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    valid_tokenized = valid_data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    
    # Remove original columns
    cols_to_remove = [col for col in train_tokenized.column_names if col != "input_ids"]
    train_tokenized = train_tokenized.remove_columns(cols_to_remove)
    valid_tokenized = valid_tokenized.remove_columns([col for col in valid_tokenized.column_names if col != "input_ids"])
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        first_key = next(iter(examples.keys()))
        total_length = len(concatenated_examples[first_key])
        # Drop the small remainder
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    train_chunked = train_tokenized.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    valid_chunked = valid_tokenized.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    
    train_chunked = train_chunked.with_format('torch')
    valid_chunked = valid_chunked.with_format('torch')
    
    return train_chunked, valid_chunked


def get_sft_dataloaders(cfg, distributed=True):
    """Create dataloaders for SFT."""
    if cfg.training.batch_size % (cfg.ngpus * cfg.training.accum) != 0:
        raise ValueError(f"Train Batch Size {cfg.training.batch_size} is not divisible by {cfg.ngpus} gpus with accumulation {cfg.training.accum}.")
    if cfg.eval.batch_size % (cfg.ngpus * cfg.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {cfg.eval.batch_size} is not divisible by {cfg.ngpus} gpus with accumulation {cfg.training.accum}.")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    max_length = cfg.sft.max_length if cfg.sft.max_length else cfg.model.length
    
    # Get optional Q&A field configurations
    question_field = cfg.sft.question_field if hasattr(cfg.sft, 'question_field') else None
    response_field = cfg.sft.response_field if hasattr(cfg.sft, 'response_field') else None
    
    train_set, valid_set = get_sft_dataset(
        cfg.sft.dataset, 
        cfg.sft.text_field, 
        tokenizer, 
        max_length,
        cache_dir=cfg.data.cache_dir,
        question_field=question_field,
        response_field=response_field
    )

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    
    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=cfg.training.batch_size // (cfg.ngpus * cfg.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader


def load_pretrained_for_sft(pretrained_path, cfg, device):
    """Load a pretrained SEDD model for fine-tuning."""
    try:
        # Try loading from HuggingFace Hub
        score_model = SEDD.from_pretrained(pretrained_path).to(device)
        graph = graph_lib.get_graph(score_model.config, device)
        noise = noise_lib.get_noise(score_model.config).to(device)
        print(f"Loaded pretrained model from HuggingFace: {pretrained_path}")
    except (OSError, ValueError, EnvironmentError):
        # Try loading from local path (OSError for missing files, ValueError for invalid model,
        # EnvironmentError for HuggingFace hub connection issues)
        local_cfg = utils.load_hydra_config_from_run(pretrained_path)
        graph = graph_lib.get_graph(local_cfg, device)
        noise = noise_lib.get_noise(local_cfg).to(device)
        score_model = SEDD(local_cfg).to(device)
        
        ckpt_dir = os.path.join(pretrained_path, "checkpoints-meta", "checkpoint.pth")
        loaded_state = torch.load(ckpt_dir, map_location=device)
        score_model.load_state_dict(loaded_state['model'])
        print(f"Loaded pretrained model from local path: {pretrained_path}")
    
    return score_model, graph, noise


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # Load pretrained model
    mprint(f"Loading pretrained model from: {cfg.sft.pretrained_model}")
    score_model, graph, noise = load_pretrained_for_sft(cfg.sft.pretrained_model, cfg, device)
    
    # Optionally freeze embeddings
    if cfg.sft.freeze_embeddings:
        mprint("Freezing embedding layers")
        for param in score_model.vocab_embed.parameters():
            param.requires_grad = False
    
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)
    noise = DDP(noise, device_ids=[rank], static_graph=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    num_trainable = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    mprint(f"Number of parameters in the model: {num_parameters}")
    mprint(f"Number of trainable parameters: {num_trainable}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    sampling_eps = 1e-5

    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 

    # load in state if resuming
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators for SFT
    mprint(f"Loading SFT dataset: {cfg.sft.dataset}")
    train_ds, eval_ds = get_sft_dataloaders(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)

    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting SFT training loop at step {initial_step}.")

    while state['step'] < num_train_steps + 1:
        step = state['step']

        batch = next(train_iter)['input_ids'].to(device)
        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                eval_batch = next(eval_iter)['input_ids'].to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = sample.shape[0] // cfg.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(batches):
                                s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                            total_perplexity /= batches
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

                            del eval_model, logits, loss

                    dist.barrier()
