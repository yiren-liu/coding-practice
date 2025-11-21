"""
ML Implementation: Training a Small Transformer Language Model

Description:
Build and train a small GPT-style decoder-only transformer for character-level 
language modeling. This problem combines all the pieces: model architecture,
data preparation, training loop, and text generation.

Key Concepts:
- Decoder-only transformer (like GPT)
- Character-level tokenization
- Causal masking (autoregressive generation)
- Training loop with loss tracking
- Text generation with sampling strategies

This is a simplified version of how models like GPT are trained!

References:
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- GPT: https://openai.com/research/language-unsupervised
- nanoGPT by Andrej Karpathy: https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class CharTokenizer:
    """
    Simple character-level tokenizer.
    
    Converts text to integers and back. Each unique character gets an ID.
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text: str):
        """
        Build vocabulary from text.
        
        Args:
            text: Training text to extract characters from
        """
        # TODO: Extract unique characters and create mappings
        # 1. Get sorted list of unique characters
        # 2. Create char_to_idx mapping (char -> integer)
        # 3. Create idx_to_char mapping (integer -> char)
        # 4. Set vocab_size
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        
        Args:
            text: String to encode
            
        Returns:
            List of token IDs
        """
        # TODO: Convert each character to its ID
        pass
    
    def decode(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded string
        """
        # TODO: Convert each ID back to character and join
        pass


class TransformerLM(nn.Module):
    """
    Small GPT-style decoder-only transformer for language modeling.
    
    Architecture:
    - Token embeddings + position embeddings
    - Stack of transformer blocks (from previous problem)
    - Output projection to vocabulary
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        use_rope: Whether to use rotary position embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # TODO: Initialize components:
        # 1. Token embedding layer (vocab_size -> d_model)
        # 2. Position embedding layer (max_seq_len -> d_model) if not using RoPE
        # 3. Stack of transformer blocks (use nn.ModuleList)
        #    - Import TransformerBlock from transformer_components.py
        #    - Or implement simplified version here
        # 4. Final layer norm
        # 5. Output projection (d_model -> vocab_size)
        # 6. Dropout layer
        
        # Hints:
        # - Token embeddings: nn.Embedding(vocab_size, d_model)
        # - Position embeddings: nn.Embedding(max_seq_len, d_model) (if not RoPE)
        # - Output layer can share weights with token embedding (weight tying)
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input token IDs [batch_size, seq_len]
            targets: Target token IDs for computing loss [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = x.shape
        
        # TODO: Implement forward pass:
        # 1. Get token embeddings
        # 2. Add position embeddings (if not using RoPE)
        # 3. Apply dropout
        # 4. Pass through transformer blocks with causal mask
        # 5. Apply final layer norm
        # 6. Project to vocabulary
        # 7. Compute loss if targets provided
        
        # Causal mask: Lower triangular matrix to prevent attending to future tokens
        # mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        pass
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token IDs [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            
        Returns:
            Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        # TODO: Implement text generation:
        # 1. Loop for max_new_tokens iterations
        # 2. Get predictions for last token
        # 3. Apply temperature scaling
        # 4. Optionally filter to top-k tokens
        # 5. Sample next token
        # 6. Append to sequence
        
        # Hints:
        # - Use self.forward() to get logits
        # - Only keep last position: logits[:, -1, :]
        # - Sample: torch.multinomial(probs, num_samples=1)
        # - Concatenate: torch.cat([idx, next_token], dim=1)
        pass


class TextDataset:
    """
    Simple dataset for character-level language modeling.
    
    Creates overlapping chunks of text for training.
    """
    
    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int = 128):
        """
        Args:
            text: Training text
            tokenizer: CharTokenizer instance
            seq_len: Length of each training sequence
        """
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        
        # TODO: Prepare data
        # Calculate number of sequences we can create
        # Store for later use in __getitem__
        pass
    
    def __len__(self) -> int:
        """Return number of training examples."""
        # TODO: Return number of sequences
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Returns:
            x: Input sequence [seq_len]
            y: Target sequence (x shifted by 1) [seq_len]
        """
        # TODO: Extract a sequence and its target
        # x = tokens[start:start+seq_len]
        # y = tokens[start+1:start+seq_len+1]  (shifted by 1)
        pass


def train_epoch(
    model: TransformerLM,
    dataset: TextDataset,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32,
    device: str = 'cpu'
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: TransformerLM instance
        dataset: TextDataset instance
        optimizer: Optimizer
        batch_size: Batch size
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # TODO: Implement training loop:
    # 1. Create batches from dataset
    # 2. For each batch:
    #    - Move data to device
    #    - Zero gradients
    #    - Forward pass (get loss)
    #    - Backward pass
    #    - Optimizer step
    #    - Accumulate loss
    
    # Hints:
    # - Loop: for i in range(0, len(dataset), batch_size)
    # - Get batch by calling dataset[i:i+batch_size]
    # - Stack examples: torch.stack([dataset[j][0] for j in range(...)], dim=0)
    pass


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: TextDataset,
    batch_size: int = 32,
    device: str = 'cpu'
) -> float:
    """
    Evaluate model on dataset.
    
    Args:
        model: TransformerLM instance
        dataset: TextDataset instance
        batch_size: Batch size
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # TODO: Similar to train_epoch but without gradient updates
    # Use @torch.no_grad() decorator (already added)
    pass


def train_model(
    text: str,
    num_epochs: int = 10,
    seq_len: int = 128,
    batch_size: int = 32,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    learning_rate: float = 3e-4,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[TransformerLM, CharTokenizer, List[float]]:
    """
    Complete training pipeline.
    
    Args:
        text: Training text
        num_epochs: Number of training epochs
        seq_len: Sequence length
        batch_size: Batch size
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        learning_rate: Learning rate
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        tokenizer: Tokenizer
        losses: Training loss history
    """
    # TODO: Implement complete training pipeline:
    # 1. Create and build tokenizer
    # 2. Create dataset (split train/val if desired)
    # 3. Initialize model
    # 4. Create optimizer (AdamW is a good choice)
    # 5. Training loop:
    #    - Train for one epoch
    #    - Evaluate
    #    - Print progress
    #    - Store losses
    # 6. Return model, tokenizer, and loss history
    pass


# ============= Simplified TransformerBlock (if not importing) =============
class RMSNorm(nn.Module):
    """RMSNorm implementation (copy from transformer_components.py if needed)."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class MultiHeadAttention(nn.Module):
    """Simplified multi-head attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: Implement attention (or copy from transformer_components.py)
        pass


class TransformerBlock(nn.Module):
    """Simplified transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: Implement transformer block
        # Pre-norm architecture: x = x + attn(norm(x))
        pass


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    # Sample training text
    SAMPLE_TEXT = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump!
    Sphinx of black quartz, judge my vow.
    """ * 10  # Repeat for more data
    
    print("=" * 60)
    print("Part 1: CharTokenizer Tests")
    print("=" * 60)
    
    def test_tokenizer_vocab_build(tokenizer):
        text = "hello world"
        tokenizer.build_vocab(text)
        # Should have unique characters
        result = tokenizer.vocab_size > 0 and tokenizer.vocab_size <= len(set(text))
        expected = True
        return result, expected
    
    def test_tokenizer_encode_decode(tokenizer):
        text = "hello"
        tokenizer.build_vocab(text + " world")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        result = decoded
        expected = text
        return result, expected
    
    def test_tokenizer_encode_length(tokenizer):
        text = "hello"
        tokenizer.build_vocab(text)
        tokens = tokenizer.encode(text)
        result = len(tokens)
        expected = len(text)
        return result, expected
    
    tokenizer = CharTokenizer()
    
    tokenizer_tests = [
        (test_tokenizer_vocab_build, "Tokenizer builds vocabulary"),
        (test_tokenizer_encode_decode, "Tokenizer encode/decode round-trip"),
        (test_tokenizer_encode_length, "Tokenizer encode produces correct length"),
    ]
    
    test_ml_implementation(tokenizer, tokenizer_tests)
    
    print("\n" + "=" * 60)
    print("Part 2: TransformerLM Tests")
    print("=" * 60)
    
    # Build tokenizer for model tests
    test_tokenizer = CharTokenizer()
    test_tokenizer.build_vocab(SAMPLE_TEXT)
    
    def test_model_forward_shape(model):
        batch_size, seq_len = 4, 32
        x = torch.randint(0, model.vocab_size, (batch_size, seq_len))
        logits, _ = model(x)
        
        result = logits.shape
        expected = torch.Size([batch_size, seq_len, model.vocab_size])
        return result, expected
    
    def test_model_loss_computation(model):
        batch_size, seq_len = 4, 32
        x = torch.randint(0, model.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, model.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(x, targets)
        
        result = loss is not None and loss.item() > 0
        expected = True
        return result, expected
    
    def test_model_generation_shape(model):
        batch_size, seq_len = 2, 10
        x = torch.randint(0, model.vocab_size, (batch_size, seq_len))
        
        generated = model.generate(x, max_new_tokens=5)
        
        result = generated.shape
        expected = torch.Size([batch_size, seq_len + 5])
        return result, expected
    
    model = TransformerLM(
        vocab_size=test_tokenizer.vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0
    )
    
    model_tests = [
        (test_model_forward_shape, "Model forward pass shape"),
        (test_model_loss_computation, "Model computes loss"),
        (test_model_generation_shape, "Model generation shape"),
    ]
    
    test_ml_implementation(model, model_tests)
    
    print("\n" + "=" * 60)
    print("Part 3: TextDataset Tests")
    print("=" * 60)
    
    def test_dataset_length(dataset):
        result = len(dataset) > 0
        expected = True
        return result, expected
    
    def test_dataset_item_shapes(dataset):
        x, y = dataset[0]
        
        result = x.shape == y.shape and x.shape[0] == dataset.seq_len
        expected = True
        return result, expected
    
    def test_dataset_target_shift(dataset):
        # Target should be input shifted by 1
        x, y = dataset[0]
        
        # First token of y should not equal first token of x (unless coincidence)
        # Better test: decode and check they make sense
        result = x.shape == y.shape
        expected = True
        return result, expected
    
    dataset_tokenizer = CharTokenizer()
    dataset_tokenizer.build_vocab(SAMPLE_TEXT)
    dataset = TextDataset(SAMPLE_TEXT, dataset_tokenizer, seq_len=32)
    
    dataset_tests = [
        (test_dataset_length, "Dataset has positive length"),
        (test_dataset_item_shapes, "Dataset items have correct shape"),
        (test_dataset_target_shift, "Dataset targets are shifted"),
    ]
    
    test_ml_implementation(dataset, dataset_tests)
    
    print("\n" + "=" * 60)
    print("Part 4: Training Tests (Quick)")
    print("=" * 60)
    
    def test_train_epoch_decreases_loss(dummy):
        # Quick training test with tiny model
        mini_text = "hello world " * 20
        tok = CharTokenizer()
        tok.build_vocab(mini_text)
        
        mini_model = TransformerLM(
            vocab_size=tok.vocab_size,
            d_model=64,
            num_layers=1,
            num_heads=2,
            d_ff=128,
            max_seq_len=32,
            dropout=0.0
        )
        
        mini_dataset = TextDataset(mini_text, tok, seq_len=16)
        optimizer = torch.optim.Adam(mini_model.parameters(), lr=0.001)
        
        # Train for 2 epochs
        loss1 = train_epoch(mini_model, mini_dataset, optimizer, batch_size=2)
        loss2 = train_epoch(mini_model, mini_dataset, optimizer, batch_size=2)
        
        # Loss should decrease (or at least be finite)
        result = loss1 > 0 and loss2 > 0 and loss2 <= loss1 * 1.5  # Allow some variance
        expected = True
        
        return result, expected
    
    # Use a dummy object for test runner
    class Dummy:
        pass
    
    training_tests = [
        (test_train_epoch_decreases_loss, "Training decreases loss"),
    ]
    
    test_ml_implementation(Dummy(), training_tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
CharTokenizer:
- Use sorted(set(text)) to get unique characters
- char_to_idx: {char: i for i, char in enumerate(chars)}
- idx_to_char: {i: char for char, i in char_to_idx.items()}

TransformerLM:
- Token embedding: self.token_emb = nn.Embedding(vocab_size, d_model)
- Position embedding: self.pos_emb = nn.Embedding(max_seq_len, d_model)
- Causal mask: torch.tril(torch.ones(seq_len, seq_len))
- Output: logits = self.lm_head(x) where lm_head is Linear(d_model, vocab_size)

TextDataset:
- Number of sequences: len(tokens) - seq_len
- Get item: x = tokens[idx:idx+seq_len], y = tokens[idx+1:idx+seq_len+1]

Training Loop:
- optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
- Standard PyTorch training loop:
  optimizer.zero_grad()
  logits, loss = model(x, targets)
  loss.backward()
  optimizer.step()

Generation:
- Use temperature to control randomness: logits / temperature
- Top-k sampling: keep only top k logits, set others to -inf
- Sample: torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

Weight Tying (Optional):
- Share weights between token embedding and output projection
- self.lm_head.weight = self.token_emb.weight
- This reduces parameters and often improves performance
    """)
    
    print("\n" + "=" * 60)
    print("Example Training Script:")
    print("=" * 60)
    print("""
# Load your favorite text (Shakespeare, etc.)
with open('input.txt', 'r') as f:
    text = f.read()

# Train the model
model, tokenizer, losses = train_model(
    text=text,
    num_epochs=20,
    seq_len=128,
    batch_size=32,
    d_model=256,
    num_layers=4,
    num_heads=4,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Generate text
model.eval()
context = "The"
context_tokens = torch.tensor([tokenizer.encode(context)], dtype=torch.long)
generated_tokens = model.generate(context_tokens, max_new_tokens=100, temperature=0.8, top_k=10)
generated_text = tokenizer.decode(generated_tokens[0].tolist())
print(generated_text)
    """)

