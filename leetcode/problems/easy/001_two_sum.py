"""
LeetCode Problem: 1 - Two Sum

Difficulty: Easy
Link: https://leetcode.com/problems/two-sum/

Description:
Given an array of integers nums and an integer target, return indices of the 
two numbers such that they add up to target.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- Only one valid answer exists.
"""

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        TODO: Implement your solution here.
        
        Time Complexity: O(?)
        Space Complexity: O(?)
        """
        pass


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    # Navigate up to leetcode directory, then to utils
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_solution
    
    solution = Solution()
    
    test_cases = [
        (([2, 7, 11, 15], 9), [0, 1], "Example 1"),
        (([3, 2, 4], 6), [1, 2], "Example 2"),
        (([3, 3], 6), [0, 1], "Example 3"),
    ]
    
    test_solution(solution.twoSum, test_cases)

