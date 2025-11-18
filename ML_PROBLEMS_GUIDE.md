# ML Coding Problems Guide

## 📚 Available Problems

### 1. Multi-Head Attention
**File:** `ml/problems/attention/multi_head_attention.py`

**What to Implement:**
- Multi-head self-attention mechanism
- Q, K, V projections and output projection
- Proper head splitting and recombination
- Causal masking support
- Batching support

**Difficulty:** ⭐⭐⭐ Medium

**Key Concepts:**
- Scaled dot-product attention
- Multiple parallel attention heads
- Linear projections
- Mask handling

**Tests Included:** 8 comprehensive tests
- Output shapes
- Causal masking
- Attention weight properties
- Batching consistency

**Expected Time:** 45-60 minutes

---

### 2. Transformer Components (RMSNorm, RoPE, Transformer Block)
**File:** `ml/problems/transformers/transformer_components.py`

**What to Implement:**

#### Part A: RMSNorm
- Root Mean Square Layer Normalization
- Simpler than LayerNorm (no mean centering)
- Used in LLaMA and modern models

#### Part B: Rotary Position Embeddings (RoPE)
- Position encoding in rotation space
- Relative position information
- Used in GPT-NeoX, PaLM, LLaMA

#### Part C: Transformer Block
- Complete transformer block with pre-norm
- Attention + Feed-forward sublayers
- Residual connections
- Optional RoPE integration

**Difficulty:** ⭐⭐⭐⭐ Hard

**Key Concepts:**
- Layer normalization variants
- Position encoding strategies
- Transformer architecture
- Pre-norm vs post-norm

**Tests Included:** 13 comprehensive tests (split across 3 components)

**Expected Time:** 90-120 minutes

---

### 3. Mixture of Experts
**File:** `ml/problems/transformers/mixture_of_experts.py`

**What to Implement:**

#### Part A: Expert Network
- Simple feed-forward expert
- Basic building block

#### Part B: Top-K Gating
- Routing mechanism
- Sparse expert selection
- Load balancing loss

#### Part C: MoE Layer
- Complete mixture of experts
- Token routing to experts
- Weighted combination of outputs
- Auxiliary loss computation

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

**Key Concepts:**
- Sparse models
- Dynamic routing
- Load balancing
- Efficient batching

**Tests Included:** 14 comprehensive tests (split across 3 components)

**Expected Time:** 120-180 minutes

---

## 🎯 Suggested Order

### Beginner Path
1. Start with `scaled_dot_product_attention.py` (already created, easier warmup)
2. Then `multi_head_attention.py` (builds on attention)
3. Then `transformer_components.py` (Part A: RMSNorm first)
4. Then `transformer_components.py` (Parts B & C)
5. Finally `mixture_of_experts.py` (most complex)

### Interview Prep Path
1. **Multi-Head Attention** - Most common interview question
2. **RMSNorm** - Quick warmup, often asked
3. **Transformer Block** - Shows system design skills
4. **RoPE** - Modern technique, shows you're up-to-date
5. **Mixture of Experts** - Advanced, impresses interviewers

---

## 🚀 How to Use

### Running Tests
```bash
# Multi-Head Attention
cd ml/problems/attention
python multi_head_attention.py

# Transformer Components
cd ml/problems/transformers
python transformer_components.py

# Mixture of Experts
cd ml/problems/transformers
python mixture_of_experts.py
```

### Expected Output (Before Implementation)
All tests will **FAIL** until you implement the solutions. This is normal!

```
Running 8 test case(s)...

✗ Output shape test raised exception: ...
✗ Causal masking test raised exception: ...
...

==================================================
Tests passed: 0/8
Tests failed: 8/8
==================================================
```

### After Implementation
```
Running 8 test case(s)...

✓ Output shape test passed (0.52ms)
✓ Causal masking test passed (1.23ms)
✓ Attention weights sum to 1 passed (0.89ms)
...

==================================================
Tests passed: 8/8
==================================================
```

---

## 💡 Implementation Tips

### Multi-Head Attention
1. **Start with projections:** Q, K, V linear layers
2. **Reshape carefully:** Pay attention to dimension ordering
3. **Use existing attention:** Can reuse scaled_dot_product_attention
4. **Test incrementally:** Get shape working first, then masking

### Transformer Components

**RMSNorm (Start Here - Easiest):**
- Just 3 lines: compute RMS, normalize, scale
- Don't overthink it!

**RoPE (Moderate):**
- Precompute sin/cos frequencies
- Apply rotation to Q and K
- The math looks scary but implementation is straightforward

**Transformer Block (Hardest):**
- Can use simplified attention or import from Problem 1
- Remember: pre-norm means normalize BEFORE sub-layer
- Don't forget residual connections!

### Mixture of Experts

**Start Simple:**
1. Implement Expert (just an FFN)
2. Implement Gate (without load balancing)
3. Implement MoE (naive loop version)
4. Add load balancing loss
5. Optimize (optional)

**Common Pitfalls:**
- Gate weights must sum to 1 per token
- Only top-k experts should have non-zero weights
- Don't forget to reshape input/output
- Load balancing prevents expert collapse

---

## 📊 Complexity Reference

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Multi-Head Attention | O(n²d) | O(n²) |
| RMSNorm | O(nd) | O(1) |
| RoPE | O(nd) | O(nd) |
| Transformer Block | O(n²d) | O(n²) |
| MoE (naive) | O(nde) | O(nd) |
| MoE (efficient) | O(nde/k) | O(nd) |

Where:
- n = sequence length
- d = model dimension
- e = number of experts
- k = top-k (experts per token)

---

## 🎓 Learning Resources

### Multi-Head Attention
- Original Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Visual Guide: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Code Guide: [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### RMSNorm
- Paper: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Used in: LLaMA, Gopher, Chinchilla

### Rotary Position Embeddings
- Paper: [RoFormer](https://arxiv.org/abs/2104.09864)
- Blog: [EleutherAI RoPE](https://blog.eleuther.ai/rotary-embeddings/)
- Used in: GPT-NeoX, LLaMA, PaLM

### Mixture of Experts
- Paper: [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Paper: [GShard](https://arxiv.org/abs/2006.16668)
- Survey: [MoE Review](https://arxiv.org/abs/2209.01667)

---

## ✅ Verification Checklist

### Multi-Head Attention
- [ ] Q, K, V projections implemented
- [ ] Multi-head split and merge working
- [ ] Causal masking works correctly
- [ ] Attention weights sum to 1
- [ ] Output projection applied
- [ ] All 8 tests pass

### Transformer Components
- [ ] RMSNorm: mean of squares ≈ 1
- [ ] RMSNorm: has learnable scale
- [ ] RoPE: different positions have different embeddings
- [ ] RoPE: works for long sequences
- [ ] Transformer: residual connections work
- [ ] Transformer: masking works
- [ ] All 13 tests pass

### Mixture of Experts
- [ ] Expert transforms input
- [ ] Gate selects exactly top-k experts
- [ ] Gate weights sum to 1
- [ ] Selected experts are valid indices
- [ ] MoE output has correct shape
- [ ] Auxiliary loss computed (if enabled)
- [ ] All 14 tests pass

---

## 🏆 Challenge Yourself

Once you've implemented the basic versions:

1. **Optimize MoE:** Implement efficient batched routing
2. **Add Flash Attention:** Speed up attention with memory-efficient implementation
3. **Implement SwiGLU:** Replace GELU with SwiGLU activation (used in LLaMA)
4. **Add KV Cache:** Implement caching for inference
5. **Group Query Attention:** Implement GQA (used in LLaMA 2)

---

## 📝 Moving to Solutions

Once you've successfully implemented and tested:

```bash
# Move to solutions directory
mv ml/problems/attention/multi_head_attention.py ml/solutions/attention/
mv ml/problems/transformers/transformer_components.py ml/solutions/transformers/
mv ml/problems/transformers/mixture_of_experts.py ml/solutions/transformers/
```

Or keep a copy in problems if you want to retry:

```bash
# Copy to solutions, keep problem version
cp ml/problems/attention/multi_head_attention.py ml/solutions/attention/
```

---

Good luck! Start with Multi-Head Attention and work your way up. These are real interview questions from top AI companies! 🚀

