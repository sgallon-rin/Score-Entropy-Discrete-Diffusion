"""
Evaluate SEDD model on AIME subset from s1K-1.1 dataset.
Calculate precision by exact match of numerical answers.
"""

import torch
import argparse
import re
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from load_model import load_model
import sampling


# Constants matching run_sft.py
THINKING_LABEL = "Thinking:"
ATTEMPT_LABEL = "Attempt:"


def extract_answer(text):
    """
    Extract numerical answer from model output.
    AIME answers are integers from 0 to 999.
    """
    if text is None:
        return None
    
    # Try to find answer after "Attempt:" label (matching SFT format)
    attempt_pattern = rf'{ATTEMPT_LABEL}\s*(. +?)(?:\n|$)'
    match = re.search(attempt_pattern, text, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        # Extract first number from the answer
        num_match = re.search(r'(\d+)', answer_text)
        if num_match: 
            return int(num_match. group(1))
    
    # Fallback: try to find \boxed{... }
    boxed_pattern = r'\\boxed\{(\d+)\}'
    match = re.search(boxed_pattern, text)
    if match:
        return int(match.group(1))
    
    # Fallback: find the last number in the text
    numbers = re.findall(r'\b(\d{1,3})\b', text)
    if numbers:
        return int(numbers[-1])
    
    return None


def normalize_answer(answer):
    """Normalize answer to integer (AIME answers are 0-999)."""
    if answer is None: 
        return None
    try:
        ans = int(float(str(answer).strip()))
        # AIME answers are between 0 and 999
        if 0 <= ans <= 999:
            return ans
        return ans % 1000  # Take mod 1000 as fallback
    except:
        return None


def load_aime_dataset(cache_dir=None):
    """Load AIME subset from s1K-1.1 dataset."""
    dataset = load_dataset("simplescaling/s1K-1.1", cache_dir=cache_dir)
    
    # Get the train split (s1K-1.1 only has train)
    if "train" in dataset:
        data = dataset["train"]
    else:
        data = dataset
    
    # Filter for AIME data
    aime_data = data.filter(lambda x: x. get('source_type', '') == 'qq8933/AIME_1983_2024')
    
    print(f"Loaded {len(aime_data)} AIME examples from s1K-1.1")
    return aime_data


def create_prompt(question):
    """Create prompt in the same format as SFT training."""
    return f"Question: {question}\n\n"


def conditional_sample(model, graph, noise, tokenizer, prompt, 
                       max_length=1024, steps=1024, batch_size=1, device='cuda'):
    """
    Perform conditional sampling with the prompt as prefix.
    Based on run_sample_cond.py logic.
    """
    # Tokenize prompt
    prefix_ids = tokenizer(prompt).input_ids
    prefix_len = len(prefix_ids)
    
    # Ensure prefix doesn't exceed max_length
    if prefix_len >= max_length:
        prefix_ids = prefix_ids[:max_length - 100]  # Leave room for generation
        prefix_len = len(prefix_ids)
    
    # Create input_ids and input_locs for projection
    input_ids = torch. tensor(prefix_ids, device=device)[None].repeat(batch_size, 1)
    input_locs = list(range(prefix_len))
    
    def proj_fun(x):
        """Project function to fix prefix tokens."""
        x[:, input_locs] = input_ids
        return x
    
    # Get sampler with projection
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, 
        (batch_size, max_length), 
        'analytic', 
        steps, 
        device=device, 
        proj_fun=proj_fun
    )
    
    # Generate samples
    samples = proj_fun(sampling_fn(model))
    
    return samples


def evaluate_aime(model_path, output_file=None, steps=1024, max_length=1024, 
                  num_samples=1, cache_dir=None, limit=None):
    """
    Evaluate SEDD model on AIME dataset.
    
    Args:
        model_path: Path to fine-tuned SEDD model
        output_file: Path to save detailed results (optional)
        steps: Number of diffusion steps for sampling
        max_length: Maximum sequence length
        num_samples: Number of samples per question (for majority voting)
        cache_dir: Cache directory for datasets
        limit: Limit number of examples to evaluate (for debugging)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model, graph, noise = load_model(model_path, device)
    model.eval()
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Load AIME dataset
    aime_data = load_aime_dataset(cache_dir)
    
    if limit:
        aime_data = aime_data.select(range(min(limit, len(aime_data))))
        print(f"Limited to {len(aime_data)} examples")
    
    # Evaluation results
    results = []
    correct = 0
    total = 0
    
    for idx, example in enumerate(tqdm(aime_data, desc="Evaluating AIME")):
        # Get question and ground truth answer
        question = example. get('question', example.get('problem', ''))
        gt_answer = example.get('answer', example.get('solution', ''))
        
        # Normalize ground truth
        gt_normalized = normalize_answer(gt_answer)
        if gt_normalized is None:
            print(f"Warning: Could not parse ground truth for example {idx}:  {gt_answer}")
            continue
        
        # Create prompt
        prompt = create_prompt(question)
        
        # Generate samples
        with torch.no_grad():
            samples = conditional_sample(
                model, graph, noise, tokenizer,
                prompt, max_length=max_length, steps=steps,
                batch_size=num_samples, device=device
            )
        
        # Decode samples
        generated_texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
        
        # Extract answers from all samples
        predicted_answers = []
        for text in generated_texts:
            # Remove the prompt from the beginning
            if text.startswith(prompt):
                response = text[len(prompt):]
            else:
                response = text
            
            pred = extract_answer(response)
            pred_normalized = normalize_answer(pred)
            predicted_answers.append(pred_normalized)
        
        # Majority voting if multiple samples
        if num_samples > 1:
            # Filter out None values
            valid_answers = [a for a in predicted_answers if a is not None]
            if valid_answers:
                from collections import Counter
                final_answer = Counter(valid_answers).most_common(1)[0][0]
            else:
                final_answer = None
        else:
            final_answer = predicted_answers[0] if predicted_answers else None
        
        # Check correctness
        is_correct = (final_answer == gt_normalized)
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        result = {
            'idx': idx,
            'question':  question[: 200] + '...' if len(question) > 200 else question,
            'ground_truth': gt_normalized,
            'predicted': final_answer,
            'all_predictions': predicted_answers,
            'correct': is_correct,
            'generated_text': generated_texts[0][:500] if generated_texts else None,
        }
        results.append(result)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(aime_data)}, "
                  f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    
    # Calculate final precision
    precision = correct / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"AIME Evaluation Results")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Total examples: {total}")
    print(f"Correct:  {correct}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print("=" * 60)
    
    # Save results if output file specified
    if output_file:
        output_data = {
            'model_path': model_path,
            'precision': precision,
            'correct': correct,
            'total': total,
            'steps': steps,
            'max_length': max_length,
            'num_samples': num_samples,
            'results': results,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    
    return precision, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEDD model on AIME dataset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned SEDD model")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for detailed results (JSON)")
    parser.add_argument("--steps", type=int, default=1024,
                        help="Number of diffusion steps for sampling")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per question (for majority voting)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for datasets")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for debugging)")
    
    args = parser.parse_args()
    
    evaluate_aime(
        model_path=args.model_path,
        output_file=args.output,
        steps=args.steps,
        max_length=args.max_length,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()