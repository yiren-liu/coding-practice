# Coding Practice Space

A well-organized workspace for practicing LeetCode problems and ML coding challenges (transformers, attention mechanisms, etc.), with **separate directories for problems and solutions**.

## 📁 Directory Structure

```
coding-practice/
├── leetcode/
│   ├── problems/             # 👈 Work on problems here
│   │   ├── easy/            
│   │   ├── medium/          
│   │   └── hard/            
│   ├── solutions/            # 👈 Completed solutions stored here
│   │   ├── easy/            
│   │   ├── medium/          
│   │   └── hard/            
│   └── utils/                # Testing utilities
│       ├── __init__.py
│       └── test_runner.py
│
├── ml/
│   ├── problems/             # 👈 Work on ML problems here
│   │   ├── attention/       
│   │   ├── transformers/    
│   │   └── neural_networks/ 
│   ├── solutions/            # 👈 Completed ML solutions stored here
│   │   ├── attention/       
│   │   ├── transformers/    
│   │   └── neural_networks/ 
│   └── utils/                # Testing utilities
│       ├── __init__.py
│       └── test_runner.py
│
├── templates/                # Templates for new problems
│   ├── leetcode_template.py
│   └── ml_template.py
│
├── requirements.txt
└── README.md
```

## 🎯 Workflow: Problems → Solutions

### The Practice Flow

1. **Start with a problem** in `leetcode/problems/` or `ml/problems/`
2. **Implement your solution** without peeking at the solutions directory
3. **Run tests** to verify your implementation
4. **Move to solutions** when complete: `mv leetcode/problems/easy/problem.py leetcode/solutions/easy/problem.py`

This keeps your workspace clean and prevents accidentally seeing solutions!

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Practice LeetCode Problems

#### Start from Template
```bash
# Copy template to problems directory
cp templates/leetcode_template.py leetcode/problems/easy/problem_name.py

# Edit and implement your solution
# Run tests
cd leetcode/problems/easy
python problem_name.py
```

#### Example: Try the Two Sum Problem
```bash
# Work on the problem (solution is removed, only test cases remain)
cd leetcode/problems/easy
python 001_two_sum.py
```

You should see test failures until you implement the solution!

#### When you solve it:
```bash
# Move to solutions directory
mv leetcode/problems/easy/001_two_sum.py leetcode/solutions/easy/001_two_sum.py

# Or keep a copy in problems if you want to retry later
cp leetcode/solutions/easy/001_two_sum.py leetcode/problems/easy/001_two_sum_v2.py
```

### 3. Practice ML Implementations

#### Start from Template
```bash
# Copy template to problems directory
cp templates/ml_template.py ml/problems/attention/your_implementation.py

# Edit and implement
cd ml/problems/attention
python your_implementation.py
```

#### Example: Try Scaled Dot-Product Attention
```bash
# Work on the attention mechanism
cd ml/problems/attention
python scaled_dot_product_attention.py
```

#### When you solve it:
```bash
# Move to solutions
mv ml/problems/attention/scaled_dot_product_attention.py ml/solutions/attention/
```

### 4. View Example Solutions

Complete, working solutions are available in the `solutions/` directories:

**LeetCode:**
```bash
cd leetcode/solutions/easy
python 001_two_sum.py
```

**ML:**
```bash
cd ml/solutions/attention
python scaled_dot_product_attention.py
```

## 📝 Creating New Problems

### Quick Create Script

```bash
# LeetCode problem
cp templates/leetcode_template.py leetcode/problems/easy/123_problem_name.py

# ML problem
cp templates/ml_template.py ml/problems/attention/multi_head_attention.py
```

### Template Features

**LeetCode Template:**
- Pre-structured problem description
- Type hints for clarity
- Built-in test runner with timing
- Easy test case format

**ML Template:**
- PyTorch/NumPy support
- Numerical comparison with tolerances
- Shape validation tests
- Computation correctness tests

## 🧪 Testing Your Solutions

### LeetCode Test Runner

```python
from utils.test_runner import test_solution

class Solution:
    def myMethod(self, nums, target):
        # Your implementation
        return result

# Define test cases
test_cases = [
    ((nums, target), expected_output, "Test name"),
]

test_solution(Solution().myMethod, test_cases)
```

**Output:**
```
Running 3 test case(s)...

✓ Example 1 passed (0.05ms)
✗ Example 2 failed
  Input: ([3, 2, 4], 6)
  Expected: [1, 2]
  Got: None
✓ Example 3 passed (0.02ms)

==================================================
Tests passed: 2/3
==================================================
```

### ML Test Runner

```python
from utils.test_runner import test_ml_implementation

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your implementation
    
    def forward(self, x):
        # Your implementation
        return x

def test_shape(model):
    x = torch.randn(2, 10, 64)
    output = model(x)
    return output.shape, (2, 10, 128)

tests = [(test_shape, "Shape test")]
test_ml_implementation(MyModel(), tests)
```

## 💡 Best Practices

### Organization Tips

1. **Name files clearly**: `001_two_sum.py`, `015_three_sum.py`
2. **Use difficulty folders**: Keep easy/medium/hard separate
3. **Don't peek at solutions**: Practice honestly for best learning
4. **Move completed work**: Keep `problems/` for active work only
5. **Document your approach**: Add comments about time/space complexity

### Learning Tips

1. **Start simple**: Begin with easy problems to build confidence
2. **Understand, don't memorize**: Focus on patterns, not specific solutions
3. **Time yourself**: Use the built-in timer to track improvement
4. **Test edge cases**: Always include boundary conditions
5. **Review solutions**: After solving, check the solutions directory for other approaches

### ML-Specific Tips

1. **Test shapes first**: Always verify tensor dimensions
2. **Use small examples**: Start with hand-calculable test cases
3. **Compare with references**: Test against PyTorch's implementations when available
4. **Document formulas**: Include LaTeX or clear mathematical notation
5. **Visualize attention**: Consider adding visualization code for debugging

## 📚 Suggested Practice Path

### LeetCode Progression

**Week 1-2: Fundamentals**
- Arrays and Strings (easy)
- Hash Maps (easy)
- Two Pointers (easy)

**Week 3-4: Core Patterns**
- Sliding Window (easy → medium)
- Binary Search (easy → medium)
- Linked Lists (easy → medium)

**Week 5-8: Advanced Topics**
- Trees and Graphs (medium)
- Dynamic Programming (medium)
- Backtracking (medium)

### ML Progression

**Week 1-2: Foundations**
- Scaled Dot-Product Attention
- Multi-Head Attention
- Layer Normalization

**Week 3-4: Transformers**
- Positional Encoding
- Transformer Encoder Block
- Transformer Decoder Block

**Week 5-6: Advanced**
- Flash Attention
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)

## 🔧 Advanced Usage

### Using pytest

Run tests with pytest for more detailed output:

```bash
# Run all tests
pytest leetcode/problems/ leetcode/solutions/

# Run specific directory
pytest ml/solutions/attention/

# Run with coverage
pytest --cov=. leetcode/ ml/
```

### Custom Test Utilities

Extend the test runners in `utils/`:

```python
class CustomTestRunner(TestRunner):
    def benchmark(self, inputs, iterations=1000):
        # Your benchmarking logic
        pass
```

## 📈 Tracking Progress

Create a `PROGRESS.md` file to track your journey:

```markdown
# Progress Log

## LeetCode
- [x] 001 - Two Sum (Easy) - 2024-01-15
- [ ] 015 - Three Sum (Medium)
- [ ] 042 - Trapping Rain Water (Hard)

## ML
- [x] Scaled Dot-Product Attention - 2024-01-15
- [ ] Multi-Head Attention
- [ ] Full Transformer

## Notes
- Pattern: Hash maps are great for O(1) lookups
- Need to review: Dynamic programming
```

## 🎓 Resources

### LeetCode
- [LeetCode Patterns](https://seanprashad.com/leetcode-patterns/)
- [NeetCode Roadmap](https://neetcode.io/roadmap)

### ML
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

Happy coding! 🚀

---

**Remember:** The best way to learn is by doing. Start with problems, struggle through them, and only check solutions when you're truly stuck or want to compare approaches.
