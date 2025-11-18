#!/bin/bash

# Helper script to create a new problem from template

print_usage() {
    echo "Usage: ./new_problem.sh [leetcode|ml] [difficulty/category] [name]"
    echo ""
    echo "Examples:"
    echo "  ./new_problem.sh leetcode easy 206_reverse_linked_list"
    echo "  ./new_problem.sh ml attention multi_head_attention"
    echo ""
    echo "LeetCode difficulties: easy, medium, hard"
    echo "ML categories: attention, transformers, neural_networks"
}

# Check arguments
if [ $# -ne 3 ]; then
    print_usage
    exit 1
fi

TYPE=$1
CATEGORY=$2
NAME=$3

# Validate type
if [ "$TYPE" != "leetcode" ] && [ "$TYPE" != "ml" ]; then
    echo "Error: Type must be 'leetcode' or 'ml'"
    print_usage
    exit 1
fi

# Set paths based on type
if [ "$TYPE" = "leetcode" ]; then
    TEMPLATE="templates/leetcode_template.py"
    TARGET="leetcode/problems/$CATEGORY/$NAME.py"
    
    # Validate difficulty
    if [ "$CATEGORY" != "easy" ] && [ "$CATEGORY" != "medium" ] && [ "$CATEGORY" != "hard" ]; then
        echo "Error: Difficulty must be 'easy', 'medium', or 'hard'"
        exit 1
    fi
else
    TEMPLATE="templates/ml_template.py"
    TARGET="ml/problems/$CATEGORY/$NAME.py"
    
    # Validate category
    if [ "$CATEGORY" != "attention" ] && [ "$CATEGORY" != "transformers" ] && [ "$CATEGORY" != "neural_networks" ]; then
        echo "Error: Category must be 'attention', 'transformers', or 'neural_networks'"
        exit 1
    fi
fi

# Check if template exists
if [ ! -f "$TEMPLATE" ]; then
    echo "Error: Template not found at $TEMPLATE"
    exit 1
fi

# Check if target already exists
if [ -f "$TARGET" ]; then
    echo "Error: File already exists at $TARGET"
    exit 1
fi

# Create the file
cp "$TEMPLATE" "$TARGET"

echo "✓ Created new problem at: $TARGET"
echo ""
echo "Next steps:"
echo "  1. Edit the file: vim $TARGET"
echo "  2. Implement your solution"
echo "  3. Run tests: python $TARGET"
echo "  4. When done, move to solutions: mv $TARGET ${TARGET/problems/solutions}"

