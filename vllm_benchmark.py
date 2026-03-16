#!/usr/bin/env python3
"""
vLLM benchmark script for remote server
Based on: https://docs.vllm.ai/en/stable/benchmarking/cli/
"""

import asyncio
import json
import os
from datetime import timedelta
from dotenv import load_dotenv
from vllm.benchmarks.serve import benchmark, get_samples, get_tokenizer, TaskType
from vllm.benchmarks.datasets import SampleRequest
from benchmark_utils import generate_markdown_summary

# Load environment variables from .env file
load_dotenv()


async def main():
    # Remote server configuration (from environment variables)
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("VLLM_API_KEY")
    
    # Validate API key
    if not api_key:
        raise ValueError("VLLM_API_KEY environment variable is not set. Please set it in your .env file.")
    
    # Benchmark parameters
    backend = "openai"  # vLLM's OpenAI-compatible API
    dataset_name = "random"  # Synthetic random dataset (no file needed)
    num_prompts = 100
    request_rate = float("inf")  # Unlimited rate (unthrottled)
    max_concurrency = None
    model_id = None  # Will be fetched from server
    model_name = None
    endpoint = "/v1/completions"
    
    api_url = f"{base_url}{endpoint}"
    
    print(f"Running benchmark against: {base_url}")
    print(f"API endpoint: {api_url}")
    print(f"Backend: {backend}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of prompts: {num_prompts}")
    print("-" * 50)
    
    # Headers with API key
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Fetch model from server if not specified
    if model_id is None:
        print("Model not specified, fetching first model from server...")
        from vllm.benchmarks.serve import get_first_model_from_server
        # get_first_model_from_server returns (id, root)
        # where id is the API-compatible model ID (e.g., "Qwen3.5-27B")
        # and root is the actual model path (e.g., "QuantTrio/Qwen3.5-27B-AWQ")
        model_id, model_root = await get_first_model_from_server(base_url, headers, None)
        model_name = model_id  # Use the API ID as the model name for requests
        print(f"Model API ID: {model_id}, Model root path: {model_root}")
    else:
        # If model_id is specified, use it for both API and tokenizer
        model_root = model_id
        model_name = model_id
    
    # Get tokenizer (required for random dataset)
    # Use the root model path for tokenizer loading
    tokenizer_id = model_root
    tokenizer_mode = "auto"
    print(f"Loading tokenizer for {tokenizer_id}...")
    tokenizer = get_tokenizer(tokenizer_id, tokenizer_mode=tokenizer_mode, trust_remote_code=False)
    
    # Create args-like object for get_samples
    class Args:
        def __init__(self):
            self.dataset_name = dataset_name
            self.num_prompts = num_prompts
            self.sharegpt_output_len = 512
            self.random_input_len = 1024  # Input length for random dataset
            self.random_output_len = 512  # Output length for random dataset
            self.random_prefix_len = 0  # Prefix length for random dataset
            self.random_range_ratio = 0.5  # Range ratio for random dataset
            self.random_batch_size = 1  # Batch size for random dataset
            self.no_oversample = False
            self.sonnet_input_len = None
            self.sonnet_output_len = None
            self.custom_output_len = None
            self.hf_output_len = None
            self.spec_bench_output_len = None
            self.prefix_repetition_output_len = None
            self.dataset_path = None
            self.random_mm_image_aspect_ratio = None
            self.random_mm_image_size = None
            self.seed = 42
            self.sharegpt_skip_prompt = False
            self.disable_shuffle = False
            self.sharegpt_prompt = True
            self.sharegpt_response = True
            
    args = Args()
    
    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    input_requests = get_samples(args, tokenizer)
    print(f"Loaded {len(input_requests)} prompts")
    
    # Run the benchmark
    benchmark_result = await benchmark(
        task_type=TaskType.GENERATION,
        endpoint_type=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=None,
        request_rate=request_rate,
        burstiness=1.0,
        disable_tqdm=False,
        num_warmups=0,
        profile=False,
        selected_percentile_metrics=["ttft", "tpot", "itl"],
        selected_percentiles=[50.0, 80.0, 90.0, 95.0, 99.0],
        ignore_eos=False,
        goodput_config_dict={},
        max_concurrency=max_concurrency,
        lora_modules=None,
        extra_headers=headers,
        extra_body={"temperature": 0.7, "top_p": 1.0},
        ramp_up_strategy=None,
        ramp_up_start_rps=None,
        ramp_up_end_rps=None,
        ready_check_timeout_sec=600,
        ssl_context=None,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    for key, value in benchmark_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results to JSON
    result_json = {
        "backend": backend,
        "model_id": model_id,
        "model_name": model_name,
        "num_prompts": num_prompts,
        "request_rate": "inf" if request_rate == float("inf") else request_rate,
        **benchmark_result
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    print(f"\nResults saved to benchmark_results.json")
    
    # Generate and save markdown summary
    generate_markdown_summary(result_json)
    print("Markdown summary saved to benchmark_summary.md")


if __name__ == "__main__":
    asyncio.run(main())
