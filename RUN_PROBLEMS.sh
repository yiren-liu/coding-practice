#!/bin/bash

# Quick runner script for ML problems
# Usage: ./RUN_PROBLEMS.sh [problem_number]

source .venv/bin/activate

echo "============================================================"
echo "ML Coding Problems - Test Runner"
echo "============================================================"
echo ""

if [ "$1" = "1" ]; then
    echo "Running Problem 1: Multi-Head Attention"
    echo "File: ml/problems/attention/multi_head_attention.py"
    echo ""
    python ml/problems/attention/multi_head_attention.py
    
elif [ "$1" = "2" ]; then
    echo "Running Problem 2: Transformer Components"
    echo "File: ml/problems/transformers/transformer_components.py"
    echo ""
    python ml/problems/transformers/transformer_components.py
    
elif [ "$1" = "3" ]; then
    echo "Running Problem 3: Mixture of Experts"
    echo "File: ml/problems/transformers/mixture_of_experts.py"
    echo ""
    python ml/problems/transformers/mixture_of_experts.py
    
else
    echo "Available problems:"
    echo ""
    echo "  1. Multi-Head Attention (Medium)"
    echo "     ./RUN_PROBLEMS.sh 1"
    echo "     File: ml/problems/attention/multi_head_attention.py"
    echo ""
    echo "  2. Transformer Components - RMSNorm, RoPE, Transformer Block (Hard)"
    echo "     ./RUN_PROBLEMS.sh 2"
    echo "     File: ml/problems/transformers/transformer_components.py"
    echo ""
    echo "  3. Mixture of Experts (Very Hard)"
    echo "     ./RUN_PROBLEMS.sh 3"
    echo "     File: ml/problems/transformers/mixture_of_experts.py"
    echo ""
    echo "Or run directly:"
    echo "  source .venv/bin/activate"
    echo "  python ml/problems/attention/multi_head_attention.py"
fi

