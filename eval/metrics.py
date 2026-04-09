import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

def calculate_perplexity(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    seq_len: int = 512,
    num_samples: int = 20, 
    device: torch.device = torch.device('cpu')
) -> float:
    """
    WHAT: Calculates true Perplexity over the Wikitext-2 test corpus.
    WHY: Evaluating on a single sentence is mathematically meaningless. We must 
    average the Negative Log-Likelihood (NLL) over a large text corpus to prove 
    the model's distribution hasn't collapsed.
    """
    print("     Downloading/Loading Wikitext-2 test data for evaluation...")
    # Load the specific test split
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Combine text and tokenize
    text_blocks = [s['text'] for s in dataset if len(s['text'].strip()) > 0]
    combined_text = " ".join(text_blocks)
    
    encodings = tokenizer(combined_text, return_tensors='pt')
    input_ids = encodings.input_ids[0]
    
    # Calculate how many full sequences we can make
    total_len = input_ids.size(0)
    nsamples = total_len // seq_len
    
    # TWEAK HERE: We limit to `num_samples` so your CPU testbed doesn't take 5 hours.
    # For your final B.Tech paper results, set num_samples to nsamples (the whole dataset).
    nsamples = min(nsamples, num_samples)
    
    nlls = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(nsamples), desc="     Evaluating"):
            # Slice the chunk
            chunk = input_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
            
            # For seq2seq models like T5, passing labels computes the loss automatically
            outputs = model(input_ids=chunk, labels=chunk)
            nlls.append(outputs.loss.item())
            
    # Average the Negative Log-Likelihoods and exponentiate to get Perplexity
    avg_nll = sum(nlls) / len(nlls)
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    
    return ppl