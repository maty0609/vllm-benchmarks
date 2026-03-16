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
- `VLLM_BASE_URL` - Remote server URL (default: `http://localhost:8000`)
- `VLLM_API_KEY` - API key for authentication (required)

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

2. **benchmark_summary.md** - Formatted markdown report with tables and analysis

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **TTFT** | Time from request to first token (latency) |
| **TPOT** | Average time per output token |
| **ITL** | Time between consecutive tokens |
| **Throughput** | Tokens generated per second |
