#!/usr/bin/env python3
"""
Utility functions for vLLM benchmarking
Contains functions that don't require vLLM dependencies
"""

from datetime import timedelta


def generate_markdown_summary(result: dict, output_path: str = "benchmark_summary.md") -> str:
    """Generate a markdown summary from benchmark results.
    
    Args:
        result: Dictionary containing benchmark results
        output_path: Path to save the markdown summary (default: benchmark_summary.md)
    
    Returns:
        The generated markdown string
    """
    model_id = result.get("model_id", "N/A")
    model_name = result.get("model_name", "N/A")
    backend = result.get("backend", "N/A")
    num_prompts = result.get("num_prompts", 0)
    completed = result.get("completed", 0)
    failed = result.get("failed", 0)
    duration = result.get("duration", 0)
    
    # TTFT metrics
    mean_ttft = result.get("mean_ttft_ms", 0)
    median_ttft = result.get("median_ttft_ms", 0)
    p50_ttft = result.get("p50_ttft_ms", 0)
    p80_ttft = result.get("p80_ttft_ms", 0)
    p90_ttft = result.get("p90_ttft_ms", 0)
    p95_ttft = result.get("p95_ttft_ms", 0)
    p99_ttft = result.get("p99_ttft_ms", 0)
    
    # TPOT metrics
    mean_tpot = result.get("mean_tpot_ms", 0)
    median_tpot = result.get("median_tpot_ms", 0)
    p50_tpot = result.get("p50_tpot_ms", 0)
    p80_tpot = result.get("p80_tpot_ms", 0)
    p90_tpot = result.get("p90_tpot_ms", 0)
    p95_tpot = result.get("p95_tpot_ms", 0)
    p99_tpot = result.get("p99_tpot_ms", 0)
    
    # ITL metrics
    mean_itl = result.get("mean_itl_ms", 0)
    median_itl = result.get("median_itl_ms", 0)
    p50_itl = result.get("p50_itl_ms", 0)
    p80_itl = result.get("p80_itl_ms", 0)
    p90_itl = result.get("p90_itl_ms", 0)
    p95_itl = result.get("p95_itl_ms", 0)
    p99_itl = result.get("p99_itl_ms", 0)
    
    output_throughput = result.get("output_throughput", 0)
    max_concurrency = result.get("max_concurrent_requests", 0)
    
    # Server configuration (optional)
    server_config = result.get("server_config", {})
    
    # Extract hardware info
    gpu_type = server_config.get("gpu_type", "N/A")
    
    # Extract model info (moved from server_config to top level)
    model_root = server_config.get("model_root", "N/A")
    
    # Extract vLLM parameters
    vllm_params = server_config.get("vllm_params", {})
    gpu_memory_utilization = vllm_params.get("gpu_memory_utilization", "N/A")
    enable_auto_tool_choice = vllm_params.get("enable_auto_tool_choice", "N/A")
    tool_call_parser = vllm_params.get("tool_call_parser", "N/A")
    reasoning_parser = vllm_params.get("reasoning_parser", "N/A")
    language_model_only = vllm_params.get("language_model_only", "N/A")
    enable_prefix_caching = vllm_params.get("enable_prefix_caching", "N/A")
    max_model_len = vllm_params.get("max_model_len", "N/A")
    enforce_eager = vllm_params.get("enforce_eager", "N/A")
    trust_remote_code = vllm_params.get("trust_remote_code", "N/A")
    
    # Format duration
    duration_td = timedelta(seconds=duration)
    duration_str = str(duration_td)
    
    # Generate summary notes
    success_rate = (completed / num_prompts * 100) if num_prompts > 0 else 0
    notes = []
    if success_rate == 100:
        notes.append(f"✅ All {completed} prompts completed successfully (0 errors)")
    else:
        notes.append(f"⚠️ {failed} out of {num_prompts} prompts failed ({100 - success_rate:.1f}% failure rate)")
    
    if mean_ttft > 10000:
        notes.append(f"⚠️ **High TTFT**: ~{mean_ttft/1000:.1f}s average - significant initial latency (likely cold start)")
    elif mean_ttft > 1000:
        notes.append(f"📊 **Moderate TTFT**: ~{mean_ttft/1000:.2f}s average")
    else:
        notes.append(f"✅ **Low TTFT**: ~{mean_ttft:.1f}ms - fast initial response")
    
    if output_throughput > 50:
        notes.append(f"✅ **Good generation speed**: ~{output_throughput:.1f} tokens/second")
    elif output_throughput > 20:
        notes.append(f"📊 **Moderate generation speed**: ~{output_throughput:.1f} tokens/second")
    else:
        notes.append(f"⚠️ **Slow generation**: ~{output_throughput:.1f} tokens/second")
    
    itl_variance = p90_itl - p50_itl if p90_itl and p50_itl else 0
    if itl_variance < 20:
        notes.append(f"✅ **Consistent ITL**: Low variance ({p50_itl:.0f}-{p90_itl:.0f}ms range)")
    else:
        notes.append(f"📊 **Variable ITL**: Higher variance ({p50_itl:.0f}-{p90_itl:.0f}ms range)")
    
    if max_concurrency > 1:
        notes.append(f"📊 Test ran with {max_concurrency} concurrent requests")
    
    notes_str = "\n".join(f"- {note}" for note in notes)
    
    # Build Hardware & Configuration section
    hardware_section = """
## 🖥️ Hardware & Configuration

### Hardware
| Parameter | Value |
|-----------|-------|
| **GPU Type** | {gpu_type} |

### vLLM Parameters
| Parameter | Value |
|-----------|-------|
| **gpu_memory_utilization** | {gpu_memory_utilization} |
| **enable_auto_tool_choice** | {enable_auto_tool_choice} |
| **tool_call_parser** | {tool_call_parser} |
| **reasoning_parser** | {reasoning_parser} |
| **language_model_only** | {language_model_only} |
| **enable_prefix_caching** | {enable_prefix_caching} |
| **max_model_len** | {max_model_len} |
| **enforce_eager** | {enforce_eager} |
| **trust_remote_code** | {trust_remote_code} |
""".format(
        gpu_type=gpu_type,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        language_model_only=language_model_only,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
    )
    
    # Build markdown
    md = f"""# 📊 Benchmark Results Summary

## Model Information
| Field | Value |
|-------|-------|
| **Model Name** | {model_name} |
| **Model Root Path** | {model_root} |
| **Backend** | {backend} |

{hardware_section}## Test Configuration
| Metric | Value |
|--------|-------|
| **Prompts Tested** | {num_prompts} |
| **Completed** | {completed} {'✅' if failed == 0 else '⚠️'} |
| **Failed** | {failed} |
| **Max Concurrent Requests** | {max_concurrency} |
| **Total Duration** | {duration:.2f} seconds ({duration_str}) |

## 🚀 Performance Metrics

### Time to First Token (TTFT)
*Latency before first token is generated*

| Percentile | Time (ms) | Time (s) |
|------------|-----------|----------|
| **Mean** | {mean_ttft:,.2f} | ~{mean_ttft/1000:.1f}s |
| **Median (p50)** | {median_ttft:,.2f} | ~{median_ttft/1000:.1f}s |
| **p80** | {p80_ttft:,.2f} | ~{p80_ttft/1000:.1f}s |
| **p90** | {p90_ttft:,.2f} | ~{p90_ttft/1000:.1f}s |
| **p95** | {p95_ttft:,.2f} | ~{p95_ttft/1000:.1f}s |
| **p99** | {p99_ttft:,.2f} | ~{p99_ttft/1000:.1f}s |

### Time Per Output Token (TPOT)
*Average time to generate each output token*

| Percentile | Time (ms) |
|------------|-----------|
| **Mean** | {mean_tpot:,.2f} |
| **Median (p50)** | {median_tpot:,.2f} |
| **p80** | {p80_tpot:,.2f} |
| **p90** | {p90_tpot:,.2f} |
| **p95** | {p95_tpot:,.2f} |
| **p99** | {p99_tpot:,.2f} |

### Inter-Token Latency (ITL)
*Time between consecutive tokens*

| Percentile | Time (ms) |
|------------|-----------|
| **Mean** | {mean_itl:,.2f} |
| **Median (p50)** | {median_itl:,.2f} |
| **p80** | {p80_itl:,.2f} |
| **p90** | {p90_itl:,.2f} |
| **p95** | {p95_itl:,.2f} |
| **p99** | {p99_itl:,.2f} |

## 📈 Throughput
| Metric | Value |
|--------|-------|
| **Output Throughput** | {output_throughput:.2f} tokens/second |

## 📝 Summary Notes
{notes_str}

---
*Generated from benchmark_results.json*
"""
    
    with open(output_path, "w") as f:
        f.write(md)
    
    return md
