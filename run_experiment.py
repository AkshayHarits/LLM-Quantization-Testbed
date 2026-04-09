import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from core.surgery import swap_linear_layers
from methods.absmax import AbsMaxQuantizer
from methods.zeropoint import ZeroPointQuantizer
from methods.nestquant.quantizer import NestQuantizer
from methods.nestquant.calibration import get_calibration_data, capture_statistics
from eval.metrics import calculate_perplexity

def main():
    print("="*60)
    print(" INITIALIZING QUANTIZATION TESTBED ")
    print("="*60)
    
    device = torch.device("cpu")
    model_name = "google/flan-t5-small"
    
    # --------------------------------------------------------
    #  MASTER CONFIGURATION (Centralized)
    # --------------------------------------------------------
    quantizer_type = "nestquant" # Options: "absmax", "zeropoint", "nestquant"
    test_ppl = True              
    
    # GLOBAL DATASET PARAMS
    SEQ_LEN = 512              # Controls both calibration and evaluation context length
    CALIB_SAMPLES = 128         # Number of blocks used to build the Hessian
    EVAL_SAMPLES = 15          # Number of blocks used to test final Perplexity
    # --------------------------------------------------------

    # 1. Load Model & Tokenizer
    print(f"\n[1] Loading tokenizer and model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # 2. Baseline Evaluation
    if test_ppl:
        print("\n[2] Running Baseline (FP32) Evaluation...")
        fp32_ppl = calculate_perplexity(model, tokenizer, seq_len=SEQ_LEN, num_samples=EVAL_SAMPLES, device=device)
        print(f"     Baseline FP32 Perplexity: {fp32_ppl:.4f}")

    # 3. Quantization Setup & Surgery
    print(f"\n[3] Starting Surgery for: {quantizer_type.upper()}")
    
    if quantizer_type == "absmax":
        quantizer = AbsMaxQuantizer(bits=4)
        q_model = swap_linear_layers(model, quantizer)
        
    elif quantizer_type == "zeropoint":
        quantizer = ZeroPointQuantizer(bits=8)
        q_model = swap_linear_layers(model, quantizer)
        
    elif quantizer_type == "nestquant":
        print("    -> Running NestQuant Calibration Pass...")
        # Use centralized params here
        calib_ids = get_calibration_data(tokenizer, num_samples=CALIB_SAMPLES, seq_len=SEQ_LEN, device=device)
        hessian_dict, x_var_dict = capture_statistics(model, calib_ids)
        
        print("    -> Initializing 4-bit E8 Lattice + QA-LDLQ...")
        quantizer = NestQuantizer(q=14, use_dither=True)
        q_model = swap_linear_layers(model, quantizer, hessian_dict=hessian_dict, x_var_dict=x_var_dict)
    else:
        raise ValueError("Invalid quantizer_type.")

    print("     Model surgery complete.")

    # 4. Quantized Evaluation
    if test_ppl:
        print("\n[4] Running Quantized Evaluation...")
        # Use centralized params here
        q_ppl = calculate_perplexity(q_model, tokenizer, seq_len=SEQ_LEN, num_samples=EVAL_SAMPLES, device=device)
        print(f"     Quantized Perplexity: {q_ppl:.4f}")

    # 5. Qualitative Translation Test
    print("\n[5] Qualitative Generation Test")
    text = "A large language model is a computational model notable for its ability to achieve general-purpose language generation."
    input_text = f"translate from english to french: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    q_outputs = q_model.generate(input_ids, max_length=50)
    q_text = tokenizer.decode(q_outputs[0], skip_special_tokens=True)
    print(f"    Input:  {text}")
    print(f"    Output: {q_text}")
    
    print("\n" + "="*60)
    print("TESTBED EXECUTION FINISHED")
    print("="*60)

if __name__ == "__main__":
    main()