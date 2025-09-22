#!/usr/bin/env python3
"""
Fix MMLU-Pro dataset compatibility issues by clearing cache and updating dependencies.
Run this once before using mmlu_pro_bench.py
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    print("🔧 Fixing MMLU-Pro compatibility issues...\n")
    
    # 1. Clear the problematic cache
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "datasets" / "TIGER-Lab___mmlu-pro",
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--TIGER-Lab--mmlu-pro",
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"📁 Found cache at: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️  Could not clear {cache_dir}: {e}")
    
    # # 2. Upgrade datasets library
    # print("\n📦 Upgrading datasets library...")
    # os.system("pip install --upgrade datasets")
    
    # 3. Test if it works now
    print("\n🧪 Testing dataset loading...")
    try:
        import datasets
        print(f"✅ Datasets version: {datasets.__version__}")
        
        # Try to load a small sample
        print("📥 Loading MMLU-Pro sample...")
        ds = datasets.load_dataset("TIGER-Lab/mmlu-pro", "default", split="test", streaming=True)
        sample = next(iter(ds))
        print(f"✅ Successfully loaded MMLU-Pro! Sample question: {sample['question'][:100]}...")
        
    except Exception as e:
        print(f"❌ Still having issues: {e}")
        print("\n💡 Alternative solution: Install lm-eval from source:")
        print("   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
        return 1
    
    print("\n✨ Fix complete! You can now run mmlu_pro_bench.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())