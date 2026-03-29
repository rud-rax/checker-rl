import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces


class Checkers6x6(AECEnv):
    metadata = {"render_modes": ["human"], "name": "checkers_6x6_v0"}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Define the two players
        self.possible_agents = ["player_0", "player_1"]
        self.render_mode = render_mode
        
        # Create 6x6 board
        self.board = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        # Set active agents
        self.agents = self.possible_agents[:]
        
        # Initialize board with pieces
        self.board = np.zeros((6, 6), dtype=np.int8)
        
        # Player 0 pieces (bottom)
        for row in range(2):
            for col in range(6):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1
        
        # Player 1 pieces (top)
        for row in range(4, 6):
            for col in range(6):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1
        
        # Initialize rewards, terminations, truncations, infos
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Agent selector for turn management
        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
    def step(self, action):
        """Execute one step - to be implemented later"""
        pass
    
    def observe(self, agent):
        """Return observation for given agent"""
        return self.board.copy()
    
    def observation_space(self, agent):
        """Define observation space"""
        return spaces.Box(low=-2, high=2, shape=(6, 6), dtype=np.int8)
    
    def action_space(self, agent):
        """Define action space - simplified for now"""
        return spaces.Discrete(64)  # Placeholder
    
    def render(self):
        """Render the board"""
        if self.render_mode == "human":
            self.print_board(self.board)
    
    def print_board(self, board):
        """Print board with chess coordinates"""
        symbols = {
            0: '·', 1: '○', 2: '⊙', -1: '●', -2: '⊗'
        }
        
        print("\n  a b c d e f")
        print("  ───────────")
        
        for row in range(5, -1, -1):
            row_str = f"{row + 1}│"
            for col in range(6):
                row_str += symbols[board[row, col]] + " "
            print(row_str)
        print()

    def is_valid_move(self, from_pos, to_pos, player):
        """
        Check if a move is valid using the 4 diagonal directions
        
        Args:
            from_pos: tuple (row, col) - starting position
            to_pos: tuple (row, col) - ending position
            player: 0 or 1 - which player is moving
        
        Returns:
            bool: True if move is valid, False otherwise
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Get the piece at starting position
        piece = self.board[from_row, from_col]
        
        # Check if piece belongs to current player
        if player == 0 and piece <= 0:
            return False
        if player == 1 and piece >= 0:
            return False
        
        # Check if destination is empty
        if self.board[to_row, to_col] != 0:
            return False
        
        is_king = abs(piece) == 2
        
        # Define possible moves: (row_offset, col_offset)
        if is_king:
            # Kings can move in all 4 diagonal directions
            possible_moves = [
                (1, 1), (1, -1),   # Forward diagonals
                (-1, 1), (-1, -1)  # Backward diagonals
            ]
        else:
            # Regular pieces move forward only
            if player == 0:  # Player 0 moves up
                possible_moves = [(1, 1), (1, -1)]
            else:  # Player 1 moves down
                possible_moves = [(-1, 1), (-1, -1)]
        
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # Check for simple move (1 square diagonally)
        if (row_diff, col_diff) in possible_moves:
            return True
        
        # Check for jump move (2 squares diagonally)
        for dr, dc in possible_moves:
            if (row_diff, col_diff) == (2*dr, 2*dc):
                # Check if there's an opponent piece in the middle
                mid_row = from_row + dr
                mid_col = from_col + dc
                mid_piece = self.board[mid_row, mid_col]
                
                # Valid jump if middle has opponent piece
                if player == 0 and mid_piece < 0:
                    return True
                if player == 1 and mid_piece > 0:
                    return True
        
        return False
        
    def get_valid_moves(self, player):
        """
        Get all valid moves for a player
        
        Args:
            player: 0 or 1 - which player
        
        Returns:
            list of tuples: [(from_pos, to_pos), ...] where each is ((from_row, from_col), (to_row, to_col))
        """
        valid_moves = []
        
        # Find all pieces belonging to the player
        for row in range(6):
            for col in range(6):
                piece = self.board[row, col]
                
                # Check if this piece belongs to current player
                if player == 0 and piece > 0:  # Player 0 pieces (1 or 2)
                    moves = self._get_piece_moves((row, col), player)
                    valid_moves.extend(moves)
                elif player == 1 and piece < 0:  # Player 1 pieces (-1 or -2)
                    moves = self._get_piece_moves((row, col), player)
                    valid_moves.extend(moves)
        
        return valid_moves


    def _get_piece_moves(self, from_pos, player):
        """
        Get all valid moves for a single piece
        
        Args:
            from_pos: tuple (row, col) - piece position
            player: 0 or 1 - which player
        
        Returns:
            list of tuples: [(from_pos, to_pos), ...]
        """
        from_row, from_col = from_pos
        piece = self.board[from_row, from_col]
        is_king = abs(piece) == 2
        
        moves = []
        
        # Define possible directions
        if is_king:
            # Kings can move in all 4 diagonal directions
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            # Regular pieces move forward only
            if player == 0:
                directions = [(1, 1), (1, -1)]  # Up
            else:
                directions = [(-1, 1), (-1, -1)]  # Down
        
        # Check each direction for simple moves and jumps
        for dr, dc in directions:
            # Simple move (1 square)
            to_row = from_row + dr
            to_col = from_col + dc
            if 0 <= to_row < 6 and 0 <= to_col < 6:
                if self.is_valid_move(from_pos, (to_row, to_col), player):
                    moves.append((from_pos, (to_row, to_col)))
            
            # Jump move (2 squares)
            to_row = from_row + 2 * dr
            to_col = from_col + 2 * dc
            if 0 <= to_row < 6 and 0 <= to_col < 6:
                if self.is_valid_move(from_pos, (to_row, to_col), player):
                    moves.append((from_pos, (to_row, to_col)))
        
        return moves





# Usage example
if __name__ == "__main__":
    env = Checkers6x6(render_mode="human")
    env.reset()
    env.render()