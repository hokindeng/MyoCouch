#!/usr/bin/env python3
"""GPU Memory Monitor for MyoCouch - Helps diagnose GPU memory issues."""

import torch
import pynvml
import time
import sys
from datetime import datetime

def format_bytes(bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return None
    
    # Initialize NVML
    pynvml.nvmlInit()
    
    device_count = torch.cuda.device_count()
    print(f"\nFound {device_count} GPU(s)\n")
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {format_bytes(info.total)}")
        print(f"  Used Memory:  {format_bytes(info.used)} ({info.used/info.total*100:.1f}%)")
        print(f"  Free Memory:  {format_bytes(info.free)} ({info.free/info.total*100:.1f}%)")
        
        # PyTorch specific memory info
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            print(f"\n  PyTorch Allocated: {format_bytes(allocated)}")
            print(f"  PyTorch Reserved:  {format_bytes(reserved)}")
            print(f"  PyTorch Free:      {format_bytes(reserved - allocated)}")
        
        # Get process information
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            print(f"\n  Running Processes:")
            for proc in processes:
                print(f"    PID {proc.pid}: {format_bytes(proc.usedGpuMemory)}")
        
        print("-" * 50)
    
    pynvml.nvmlShutdown()

def monitor_memory(interval=1, duration=10):
    """Monitor GPU memory usage over time."""
    print(f"Monitoring GPU memory for {duration} seconds...")
    print("Time\t\tUsed\t\tFree\t\tPyTorch Allocated")
    print("-" * 60)
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        allocated = torch.cuda.memory_allocated(0) if torch.cuda.is_initialized() else 0
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp}\t{format_bytes(info.used)}\t{format_bytes(info.free)}\t{format_bytes(allocated)}")
        
        time.sleep(interval)
    
    pynvml.nvmlShutdown()

def test_allocation(size_gb=1):
    """Test allocating a specific amount of GPU memory."""
    print(f"\nTesting allocation of {size_gb} GB...")
    
    try:
        # Try to allocate memory
        size_bytes = int(size_gb * 1024 * 1024 * 1024)
        tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device='cuda')
        print(f"✓ Successfully allocated {size_gb} GB")
        
        # Check fragmentation
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"  Allocated: {format_bytes(allocated)}")
        print(f"  Reserved:  {format_bytes(reserved)}")
        print(f"  Overhead:  {format_bytes(reserved - allocated)} ({(reserved-allocated)/allocated*100:.1f}%)")
        
        # Clean up
        del tensor
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"✗ Failed to allocate {size_gb} GB: {str(e)}")
        
        # Try to find maximum allocatable size
        print("\nFinding maximum allocatable size...")
        max_size = 0
        for test_gb in [0.1, 0.5, 1, 2, 4, 8, 10, 12]:
            try:
                test_bytes = int(test_gb * 1024 * 1024 * 1024)
                test_tensor = torch.empty(test_bytes // 4, dtype=torch.float32, device='cuda')
                max_size = test_gb
                del test_tensor
                torch.cuda.empty_cache()
            except:
                break
        
        print(f"Maximum allocatable: ~{max_size} GB")

def clear_cache():
    """Clear PyTorch's GPU cache."""
    print("\nClearing PyTorch GPU cache...")
    
    before = torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    after = torch.cuda.memory_reserved()
    
    print(f"Freed: {format_bytes(before - after)}")

def main():
    """Main function."""
    print("MyoCouch GPU Memory Monitor")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            monitor_memory(duration=duration)
        elif command == "test":
            size = float(sys.argv[2]) if len(sys.argv) > 2 else 1
            test_allocation(size)
        elif command == "clear":
            clear_cache()
        else:
            print(f"Unknown command: {command}")
    else:
        # Default: show current status
        get_gpu_memory_info()
        
        print("\nUsage:")
        print("  python gpu_monitor.py           # Show current GPU memory status")
        print("  python gpu_monitor.py monitor [seconds]  # Monitor memory over time")
        print("  python gpu_monitor.py test [GB]  # Test allocating specific amount")
        print("  python gpu_monitor.py clear      # Clear PyTorch cache")

if __name__ == "__main__":
    main() 