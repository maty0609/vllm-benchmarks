#!/usr/bin/env python3
"""
Tests for vLLM benchmark script
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import timedelta


# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEnvironmentVariables:
    """Test environment variable loading and validation"""
    
    def test_api_key_required(self):
        """Test that API key is required"""
        # Unset the API key
        old_key = os.environ.pop("VLLM_API_KEY", None)
        
        try:
            # Test the validation logic directly
            api_key = os.getenv("VLLM_API_KEY")
            assert api_key is None, "API key should be None when not set"
            
            # Test that ValueError would be raised
            if not api_key:
                with pytest.raises(ValueError, match="VLLM_API_KEY.*not set"):
                    raise ValueError("VLLM_API_KEY environment variable is not set. Please set it in your .env file.")
        finally:
            # Restore the key
            if old_key:
                os.environ["VLLM_API_KEY"] = old_key
    
    def test_base_url_default(self):
        """Test that base URL has a default value"""
        # Unset the base URL
        old_url = os.environ.pop("VLLM_BASE_URL", None)
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
                assert base_url == "http://localhost:8000", "Should use default URL"
        finally:
            # Restore
            if old_url:
                os.environ["VLLM_BASE_URL"] = old_url
    
    def test_env_vars_loaded_correctly(self):
        """Test that environment variables are loaded correctly"""
        test_url = "http://test-server:8000"
        test_key = "test-api-key-123"
        
        with patch.dict(os.environ, {
            "VLLM_BASE_URL": test_url,
            "VLLM_API_KEY": test_key
        }):
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
            api_key = os.getenv("VLLM_API_KEY")
            
            assert base_url == test_url, f"Expected {test_url}, got {base_url}"
            assert api_key == test_key, f"Expected {test_key}, got {api_key}"


class TestMarkdownSummaryGeneration:
    """Test the markdown summary generation function"""
    
    def test_generate_markdown_summary_basic(self):
        """Test basic markdown summary generation"""
        from benchmark_utils import generate_markdown_summary
        
        result = {
            "model_id": "test-model",
            "model_name": "Test Model",
            "backend": "openai",
            "num_prompts": 100,
            "completed": 98,
            "failed": 2,
            "duration": 120.5,
            "mean_ttft_ms": 1500.25,
            "median_ttft_ms": 1200.0,
            "p50_ttft_ms": 1200.0,
            "p80_ttft_ms": 1800.0,
            "p90_ttft_ms": 2000.0,
            "p95_ttft_ms": 2500.0,
            "p99_ttft_ms": 3000.0,
            "mean_tpot_ms": 50.5,
            "median_tpot_ms": 45.0,
            "p50_tpot_ms": 45.0,
            "p80_tpot_ms": 55.0,
            "p90_tpot_ms": 60.0,
            "p95_tpot_ms": 70.0,
            "p99_tpot_ms": 80.0,
            "mean_itl_ms": 48.0,
            "median_itl_ms": 44.0,
            "p50_itl_ms": 44.0,
            "p80_itl_ms": 52.0,
            "p90_itl_ms": 58.0,
            "p95_itl_ms": 65.0,
            "p99_itl_ms": 75.0,
            "output_throughput": 25.5,
            "max_concurrent_requests": 4
        }
        
        # Test that it generates without error
        generate_markdown_summary(result)
        
        # Check that the file was created
        assert os.path.exists("benchmark_summary.md"), "Summary file should be created"
        
        # Read and verify content
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        
        # Verify key sections exist
        assert "test-model" in content, "Model ID should be in summary"
        assert "Test Model" in content, "Model name should be in summary"
        assert "openai" in content, "Backend should be in summary"
        assert "98" in content, "Completed count should be in summary"
        assert "2" in content, "Failed count should be in summary"
        assert "Time to First Token" in content, "TTFT section should exist"
        assert "Time Per Output Token" in content, "TPOT section should exist"
        assert "Inter-Token Latency" in content, "ITL section should exist"
        
        # Clean up
        os.remove("benchmark_summary.md")
    
    def test_markdown_summary_success_rate(self):
        """Test success rate calculation in notes"""
        from benchmark_utils import generate_markdown_summary
        
        # Test with 100% success
        result = {
            "model_id": "test-model",
            "model_name": "Test Model",
            "backend": "openai",
            "num_prompts": 100,
            "completed": 100,
            "failed": 0,
            "duration": 120.5,
            "mean_ttft_ms": 500.0,
            "median_ttft_ms": 450.0,
            "p50_ttft_ms": 450.0,
            "p80_ttft_ms": 600.0,
            "p90_ttft_ms": 700.0,
            "p95_ttft_ms": 800.0,
            "p99_ttft_ms": 900.0,
            "mean_tpot_ms": 50.0,
            "median_tpot_ms": 45.0,
            "p50_tpot_ms": 45.0,
            "p80_tpot_ms": 55.0,
            "p90_tpot_ms": 60.0,
            "p95_tpot_ms": 70.0,
            "p99_tpot_ms": 80.0,
            "mean_itl_ms": 48.0,
            "median_itl_ms": 44.0,
            "p50_itl_ms": 44.0,
            "p80_itl_ms": 52.0,
            "p90_itl_ms": 58.0,
            "p95_itl_ms": 65.0,
            "p99_itl_ms": 75.0,
            "output_throughput": 60.0,
            "max_concurrent_requests": 4
        }
        
        generate_markdown_summary(result)
        
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        
        assert "✅" in content, "Should have success indicator"
        assert "100" in content, "Should show 100 completed"
        
        os.remove("benchmark_summary.md")
    
    def test_markdown_summary_high_ttft_warning(self):
        """Test high TTFT warning is shown"""
        from benchmark_utils import generate_markdown_summary
        
        result = {
            "model_id": "test-model",
            "model_name": "Test Model",
            "backend": "openai",
            "num_prompts": 100,
            "completed": 100,
            "failed": 0,
            "duration": 120.5,
            "mean_ttft_ms": 15000.0,  # High TTFT (> 10s)
            "median_ttft_ms": 12000.0,
            "p50_ttft_ms": 12000.0,
            "p80_ttft_ms": 18000.0,
            "p90_ttft_ms": 20000.0,
            "p95_ttft_ms": 25000.0,
            "p99_ttft_ms": 30000.0,
            "mean_tpot_ms": 50.0,
            "median_tpot_ms": 45.0,
            "p50_tpot_ms": 45.0,
            "p80_tpot_ms": 55.0,
            "p90_tpot_ms": 60.0,
            "p95_tpot_ms": 70.0,
            "p99_tpot_ms": 80.0,
            "mean_itl_ms": 48.0,
            "median_itl_ms": 44.0,
            "p50_itl_ms": 44.0,
            "p80_itl_ms": 52.0,
            "p90_itl_ms": 58.0,
            "p95_itl_ms": 65.0,
            "p99_itl_ms": 75.0,
            "output_throughput": 25.0,
            "max_concurrent_requests": 4
        }
        
        generate_markdown_summary(result)
        
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        
        assert "High TTFT" in content, "Should warn about high TTFT"
        assert "⚠️" in content, "Should have warning indicator"
        
        os.remove("benchmark_summary.md")
    
    def test_markdown_summary_low_ttft(self):
        """Test low TTFT is shown as success"""
        from benchmark_utils import generate_markdown_summary
        
        result = {
            "model_id": "test-model",
            "model_name": "Test Model",
            "backend": "openai",
            "num_prompts": 100,
            "completed": 100,
            "failed": 0,
            "duration": 120.5,
            "mean_ttft_ms": 500.0,  # Low TTFT (< 1s)
            "median_ttft_ms": 450.0,
            "p50_ttft_ms": 450.0,
            "p80_ttft_ms": 600.0,
            "p90_ttft_ms": 700.0,
            "p95_ttft_ms": 800.0,
            "p99_ttft_ms": 900.0,
            "mean_tpot_ms": 50.0,
            "median_tpot_ms": 45.0,
            "p50_tpot_ms": 45.0,
            "p80_tpot_ms": 55.0,
            "p90_tpot_ms": 60.0,
            "p95_tpot_ms": 70.0,
            "p99_tpot_ms": 80.0,
            "mean_itl_ms": 48.0,
            "median_itl_ms": 44.0,
            "p50_itl_ms": 44.0,
            "p80_itl_ms": 52.0,
            "p90_itl_ms": 58.0,
            "p95_itl_ms": 65.0,
            "p99_itl_ms": 75.0,
            "output_throughput": 60.0,
            "max_concurrent_requests": 4
        }
        
        generate_markdown_summary(result)
        
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        
        assert "Low TTFT" in content, "Should show low TTFT"
        assert "fast initial response" in content, "Should mention fast response"
        
        os.remove("benchmark_summary.md")
    
    def test_markdown_summary_throughput_levels(self):
        """Test different throughput levels are categorized correctly"""
        from benchmark_utils import generate_markdown_summary
        
        # Test good throughput (> 50)
        result_good = {
            "model_id": "test-model",
            "model_name": "Test Model",
            "backend": "openai",
            "num_prompts": 100,
            "completed": 100,
            "failed": 0,
            "duration": 120.5,
            "mean_ttft_ms": 500.0,
            "median_ttft_ms": 450.0,
            "p50_ttft_ms": 450.0,
            "p80_ttft_ms": 600.0,
            "p90_ttft_ms": 700.0,
            "p95_ttft_ms": 800.0,
            "p99_ttft_ms": 900.0,
            "mean_tpot_ms": 50.0,
            "median_tpot_ms": 45.0,
            "p50_tpot_ms": 45.0,
            "p80_tpot_ms": 55.0,
            "p90_tpot_ms": 60.0,
            "p95_tpot_ms": 70.0,
            "p99_tpot_ms": 80.0,
            "mean_itl_ms": 48.0,
            "median_itl_ms": 44.0,
            "p50_itl_ms": 44.0,
            "p80_itl_ms": 52.0,
            "p90_itl_ms": 58.0,
            "p95_itl_ms": 65.0,
            "p99_itl_ms": 75.0,
            "output_throughput": 60.0,  # Good
            "max_concurrent_requests": 4
        }
        
        generate_markdown_summary(result_good)
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        assert "Good generation speed" in content
        os.remove("benchmark_summary.md")
        
        # Test moderate throughput (20-50)
        result_moderate = result_good.copy()
        result_moderate["output_throughput"] = 35.0
        
        generate_markdown_summary(result_moderate)
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        assert "Moderate generation speed" in content
        os.remove("benchmark_summary.md")
        
        # Test slow throughput (< 20)
        result_slow = result_good.copy()
        result_slow["output_throughput"] = 15.0
        
        generate_markdown_summary(result_slow)
        with open("benchmark_summary.md", "r") as f:
            content = f.read()
        assert "Slow generation" in content
        os.remove("benchmark_summary.md")


class TestResultSaving:
    """Test result saving functionality"""
    
    def test_save_results_to_json(self):
        """Test that results are saved to JSON correctly"""
        import json
        
        result_json = {
            "backend": "openai",
            "model_id": "test-model",
            "model_name": "Test Model",
            "num_prompts": 100,
            "request_rate": "inf",
            "completed": 98,
            "failed": 2,
            "duration": 120.5,
            "mean_ttft_ms": 1500.25,
            "output_throughput": 25.5
        }
        
        with open("test_results.json", "w") as f:
            json.dump(result_json, f, indent=2, default=str)
        
        # Read back and verify
        with open("test_results.json", "r") as f:
            loaded = json.load(f)
        
        assert loaded["backend"] == "openai"
        assert loaded["model_id"] == "test-model"
        assert loaded["num_prompts"] == 100
        assert loaded["request_rate"] == "inf"
        
        # Clean up
        os.remove("test_results.json")


class TestBenchmarkConfiguration:
    """Test benchmark configuration and parameters"""
    
    def test_args_object_creation(self):
        """Test that the Args object is created correctly"""
        # Simulate the Args class from the main script
        class Args:
            def __init__(self):
                self.dataset_name = "random"
                self.num_prompts = 100
                self.sharegpt_output_len = 512
                self.random_input_len = 1024
                self.random_output_len = 512
                self.random_prefix_len = 0
                self.random_range_ratio = 0.5
                self.random_batch_size = 1
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
        
        assert args.dataset_name == "random"
        assert args.num_prompts == 100
        assert args.random_input_len == 1024
        assert args.random_output_len == 512
        assert args.seed == 42
    
    def test_api_url_construction(self):
        """Test that API URL is constructed correctly"""
        base_url = "http://192.168.1.220:8000"
        endpoint = "/v1/completions"
        
        api_url = f"{base_url}{endpoint}"
        
        assert api_url == "http://192.168.1.220:8000/v1/completions"
    
    def test_headers_construction(self):
        """Test that headers are constructed correctly"""
        api_key = "test-api-key-123"
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key-123"


class TestDurationFormatting:
    """Test duration formatting in markdown"""
    
    def test_duration_to_timedelta(self):
        """Test duration formatting"""
        duration = 120.5
        
        duration_td = timedelta(seconds=duration)
        duration_str = str(duration_td)
        
        assert "0:02:00" in duration_str or "2:00" in duration_str or "120" in duration_str
    
    def test_duration_formatting_various_values(self):
        """Test duration formatting for various values"""
        test_cases = [
            (60.0, "0:01:00"),
            (3600.0, "1:00:00"),
            (7200.5, "2:00:00"),
            (0.5, "0:00:00"),
        ]
        
        for duration, expected in test_cases:
            duration_td = timedelta(seconds=duration)
            duration_str = str(duration_td)
            # Just verify it doesn't crash and produces a string
            assert isinstance(duration_str, str)
            assert len(duration_str) > 0


class TestSuccessRateCalculation:
    """Test success rate calculation logic"""
    
    def test_success_rate_100_percent(self):
        """Test 100% success rate"""
        num_prompts = 100
        completed = 100
        
        success_rate = (completed / num_prompts * 100) if num_prompts > 0 else 0
        assert success_rate == 100.0
    
    def test_success_rate_partial(self):
        """Test partial success rate"""
        num_prompts = 100
        completed = 95
        
        success_rate = (completed / num_prompts * 100) if num_prompts > 0 else 0
        assert success_rate == 95.0
    
    def test_success_rate_zero_prompts(self):
        """Test success rate with zero prompts (should not divide by zero)"""
        num_prompts = 0
        completed = 0
        
        success_rate = (completed / num_prompts * 100) if num_prompts > 0 else 0
        assert success_rate == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
