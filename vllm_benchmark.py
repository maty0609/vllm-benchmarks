#!/usr/bin/env python3
"""
vLLM benchmark script for remote server
Based on: https://docs.vllm.ai/en/stable/benchmarking/cli/
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from vllm.benchmarks.serve import benchmark, get_samples, get_tokenizer, TaskType
from vllm.benchmarks.datasets import SampleRequest
from benchmark_utils import generate_markdown_summary

# Load environment variables from .env file
load_dotenv()


async def fetch_server_config(base_url: str, headers: dict) -> dict:
    """Fetch server configuration from vLLM API endpoints and environment variables.
    
    Returns dict with model info, GPU info, and vLLM configuration parameters.
    Gracefully handles missing endpoints with empty values or environment variable fallbacks.
    """
    import aiohttp
    
    config = {}
    connector = aiohttp.TCPConnector()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Fetch full model info from /v1/models
        try:
            async with session.get(f"{base_url}/v1/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and len(data["data"]) > 0:
                        model_info = data["data"][0]
                        config["model_id"] = model_info.get("id", "N/A")
                        config["model_root"] = model_info.get("root", "N/A")
                        config["model_object"] = model_info.get("object", "model")
                else:
                    print(f"Warning: /v1/models returned status {response.status}")
        except Exception as e:
            print(f"Warning: Could not fetch model info from /v1/models: {e}")
        
        # 2. Fetch server metrics from /metrics (Prometheus format)
        try:
            async with session.get(f"{base_url}/metrics", headers=headers) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    # Parse key metrics from Prometheus format
                    config["metrics"] = parse_vllm_metrics(metrics_text)
                else:
                    print(f"Warning: /metrics returned status {response.status}")
        except Exception as e:
            print(f"Warning: Could not fetch server metrics from /metrics: {e}")
    
    # 3. Add GPU type from environment variable (vLLM doesn't expose this via API)
    gpu_type = os.getenv("GPU_TYPE")
    if gpu_type:
        config["gpu_type"] = gpu_type
    else:
        config["gpu_type"] = "Not specified (set GPU_TYPE env variable)"
    
    # 4. Add vLLM configuration parameters from environment variables
    # These are not exposed via vLLM API, so we use env vars as fallback
    vllm_params = {
        "gpu_memory_utilization": os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95"),
        "enable_auto_tool_choice": os.getenv("VLLM_ENABLE_AUTO_TOOL_CHOICE", "true"),
        "tool_call_parser": os.getenv("VLLM_TOOL_CALL_PARSER", "qwen3_coder"),
        "reasoning_parser": os.getenv("VLLM_REASONING_PARSER", "qwen3"),
        "language_model_only": os.getenv("VLLM_LANGUAGE_MODEL_ONLY", "true"),
        "enable_prefix_caching": os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true"),
        "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
        "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "true"),
        "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "true"),
    }
    config["vllm_params"] = vllm_params
    
    return config


def parse_vllm_metrics(metrics_text: str) -> dict:
    """Parse Prometheus metrics text to extract key vLLM configuration.
    
    Extracts: GPU memory usage, cache usage, block counts, request counts.
    """
    parsed = {}
    
    for line in metrics_text.split('\n'):
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        
        # Parse metric lines: metric_name{labels} value
        if '{' in line:
            parts = line.split('{')
            metric_name = parts[0].strip()
            value = parts[1].split('}')[-1].strip() if len(parts) > 1 else ""
            
            # Extract specific metrics we care about
            if metric_name == "vllm:gpu_memory_usage_gb":
                parsed["gpu_memory_usage_gb"] = float(value) if value else 0
            elif metric_name == "vllm:gpu_memory_usage":
                # Convert bytes to GB
                parsed["gpu_memory_usage_gb"] = float(value) / (1024**3) if value else 0
            elif metric_name == "vllm:gpu_cache_usage_perc":
                parsed["gpu_cache_usage"] = float(value) if value else 0
            elif metric_name == "vllm:num_gpu_blocks":
                parsed["num_gpu_blocks"] = int(float(value)) if value else 0
            elif metric_name == "vllm:num_cpu_blocks":
                parsed["num_cpu_blocks"] = int(float(value)) if value else 0
            elif metric_name == "vllm:num_requests_running":
                parsed["running_requests"] = int(float(value)) if value else 0
            elif metric_name == "vllm:num_requests_waiting":
                parsed["waiting_requests"] = int(float(value)) if value else 0
            elif metric_name == "vllm:gpu_utilization":
                parsed["gpu_utilization"] = float(value) if value else 0
    
    return parsed


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
    
    # Fetch server configuration
    print("\nFetching server configuration...")
    server_config = await fetch_server_config(base_url, headers)
    print(f"GPU Type: {server_config.get('gpu_type', 'N/A')}")
    print(f"Model Root: {server_config.get('model_root', 'N/A')}")
    if server_config.get('metrics'):
        metrics = server_config['metrics']
        if metrics.get('gpu_memory_usage_gb'):
            print(f"GPU Memory Usage: {metrics['gpu_memory_usage_gb']:.2f} GB")
        if metrics.get('gpu_cache_usage'):
            print(f"GPU Cache Usage: {metrics['gpu_cache_usage']:.1f}%")
    print("-" * 50)
    
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
    
    # Save results to JSON with organized folder structure and timestamped filename
    result_json = {
        "backend": backend,
        "model_id": model_id,
        "model_name": model_name,
        "num_prompts": num_prompts,
        "request_rate": "inf" if request_rate == float("inf") else request_rate,
        **benchmark_result
    }
    
    # Add server configuration to results
    result_json["server_config"] = server_config
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_id)
    
    # Create result folder if it doesn't exist
    os.makedirs("result", exist_ok=True)
    
    # Save JSON results
    result_filename = f"result/benchmark_{safe_model_name}_{timestamp}.json"
    with open(result_filename, "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    
    # Generate and save markdown summary
    os.makedirs("summary", exist_ok=True)
    summary_filename = f"summary/benchmark_{safe_model_name}_{timestamp}.md"
    summary_md = generate_markdown_summary(result_json, summary_filename)
    
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(summary_md)
    print(f"\nResults saved to {result_filename}")
    print(f"Markdown summary saved to {summary_filename}")


if __name__ == "__main__":
    asyncio.run(main())
