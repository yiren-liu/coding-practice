# Quick Start Guide

## TL;DR

**Work on problems** → **Test** → **Move to solutions when done**

```bash
# LeetCode
leetcode/problems/    # Work here (no solutions)
leetcode/solutions/   # Complete solutions stored here

# ML
ml/problems/          # Work here (no solutions)
ml/solutions/         # Complete solutions stored here
```

## 🎯 Typical Workflow

### LeetCode Problem

```bash
# 1. Copy template to problems
cp templates/leetcode_template.py leetcode/problems/easy/206_reverse_linked_list.py

# 2. Edit the file, add problem description and implement
vim leetcode/problems/easy/206_reverse_linked_list.py

# 3. Run tests while developing
python leetcode/problems/easy/206_reverse_linked_list.py

# 4. When solved, move to solutions
mv leetcode/problems/easy/206_reverse_linked_list.py leetcode/solutions/easy/
```

### ML Implementation

```bash
# 1. Copy template to problems
cp templates/ml_template.py ml/problems/attention/multi_head_attention.py

# 2. Edit and implement
vim ml/problems/attention/multi_head_attention.py

# 3. Run tests
python ml/problems/attention/multi_head_attention.py

# 4. When working, move to solutions
mv ml/problems/attention/multi_head_attention.py ml/solutions/attention/
```

## 📝 Try the Examples

### Example 1: LeetCode Two Sum (Easy)

**Try the problem (should fail until you implement):**
```bash
python leetcode/problems/easy/001_two_sum.py
```

Output: All tests fail ✗

**Check the solution:**
```bash
python leetcode/solutions/easy/001_two_sum.py
```

Output: All tests pass ✓

### Example 2: Scaled Dot-Product Attention

**Try the problem:**
```bash
# Install dependencies first
pip install -r requirements.txt

# Then try the problem
python ml/problems/attention/scaled_dot_product_attention.py
```

**Check the solution:**
```bash
python ml/solutions/attention/scaled_dot_product_attention.py
```

## 🔥 Pro Tips

1. **Never look at solutions first** - Practice honestly for better learning
2. **Keep problems/ clean** - Move completed work to solutions/
3. **Use descriptive names** - `001_two_sum.py` is better than `problem1.py`
4. **Test frequently** - Run tests after every small change
5. **Document complexity** - Always note time/space complexity

## 🎓 Suggested First Problems

### LeetCode (Easy)
- `001_two_sum.py` (already has starter code!)
- Two Pointers problems
- Hash Map problems

### ML
- `scaled_dot_product_attention.py` (already has starter!)
- Layer Normalization
- Positional Encoding

## 📚 Full Documentation

See [README.md](README.md) for complete documentation, tips, and resources.

---

**Ready to start?** Pick a problem and start coding! 🚀

