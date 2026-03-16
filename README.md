# vllm-benchmarks

vLLM benchmarking tools for remote server performance testing.

## Files

- **vllm_benchmark.py** - Main benchmark script using vLLM's built-in benchmarking
- **.env.example** - Template for environment variables
- **benchmark_results.json** - Raw benchmark results (JSON format)
- **benchmark_summary.md** - Human-readable markdown summary of results

## Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   ```
   VLLM_BASE_URL="http://your-server:8000"
   VLLM_API_KEY="your-api-key-here"
   ```

   ⚠️ **Important**: The `.env` file is gitignored. Never commit it with real credentials!

## Usage

```bash
# Run benchmark against remote vLLM server
python3 vllm_benchmark.py
```

## Configuration

Set environment variables in `.env` file:

### Required
- `VLLM_BASE_URL` - Remote server URL (default: `http://localhost:8000`)
- `VLLM_API_KEY` - API key for authentication (required)

### Optional - Hardware & Configuration
- `GPU_TYPE` - GPU model name (e.g., "NVIDIA A100 80GB", "NVIDIA H100")
  - Note: vLLM doesn't expose GPU model via API, so this must be set manually

### Optional - vLLM Server Parameters
These parameters are displayed in the benchmark summary for reference.
Update them to match your actual vLLM server configuration:
- `VLLM_GPU_MEMORY_UTILIZATION` - GPU memory utilization (default: "0.95")
- `VLLM_ENABLE_AUTO_TOOL_CHOICE` - Auto tool choice (default: "true")
- `VLLM_TOOL_CALL_PARSER` - Tool call parser (default: "qwen3_coder")
- `VLLM_REASONING_PARSER` - Reasoning parser (default: "qwen3")
- `VLLM_LANGUAGE_MODEL_ONLY` - Language model only mode (default: "true")
- `VLLM_ENABLE_PREFIX_CACHING` - Prefix caching (default: "true")
- `VLLM_MAX_MODEL_LEN` - Max model length in tokens (default: "131072")
- `VLLM_ENFORCE_EAGER` - Enforce eager mode (default: "true")
- `VLLM_TRUST_REMOTE_CODE` - Trust remote code (default: "true")

Or edit `vllm_benchmark.py` for other parameters:
- `num_prompts` - Number of prompts to test (default: 100)
- `max_concurrency` - Maximum concurrent requests

## Output

After running, you'll get:
1. **benchmark_results.json** - Complete metrics including:
   - TTFT (Time to First Token)
   - TPOT (Time Per Output Token)
   - ITL (Inter-Token Latency)
   - Percentiles (p50, p80, p90, p95, p99)
   - Throughput metrics
   - Server configuration (GPU type, vLLM parameters)

2. **benchmark_summary.md** - Formatted markdown report with:
   - Model Information
   - **Hardware & Configuration** (GPU type, memory usage, vLLM parameters)
   - Test Configuration
   - Performance Metrics (TTFT, TPOT, ITL)
   - Throughput
   - Summary Notes and Analysis

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **TTFT** | Time from request to first token (latency) |
| **TPOT** | Average time per output token |
| **ITL** | Time between consecutive tokens |
| **Throughput** | Tokens generated per second |
