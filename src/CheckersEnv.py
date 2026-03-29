

from typing import Optional
import numpy as np
import gymnasium as gym

from pettingzoo import AECEnv


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 6):
        # The size of the square grid (5x5 by default)
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=np.int8)

        self.__init_pieces()

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 1]),   # up right
            1: np.array([-1, 1]),  # up left
            2: np.array([-1, -1]),   # down left (for king)
            3: np.array([1, -1]),  # down right (for king)
        }

    def get(self) :
        return self.board
    
    def __init_pieces(self) :
        # Player 0 pieces (bottom two rows) - value 1
        # Only on dark squares where (row + col) is odd
        for row in range(2):
            for col in range(6):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1

        # Player 1 pieces (top two rows) - value -1
        # Only on dark squares where (row + col) is odd
        for row in range(4, 6):
            for col in range(6):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1
    
    def print_board(self):
        """Print board"""
        
        # Piece symbols
        symbols = {
            0: '·',   # Empty
            1: '○',   # Player 0 piece
            2: '⊙',   # Player 0 king
            -1: '●',  # Player 1 piece
            -2: '⊗',  # Player 1 king
        }
        
        # Column labels (a-f)
        print("\n  a b c d e f")
        print("  " + "─" * 11)
        
        # Print rows from 6 down to 1 (chess style)
        for row in range(5, -1, -1):
            row_str = f"{row + 1}│"  # Row numbers 1-6
            for col in range(6):
                piece = self.board[row, col]
                row_str += symbols[piece] + " "
            print(row_str)
        
        print()

        

        
        