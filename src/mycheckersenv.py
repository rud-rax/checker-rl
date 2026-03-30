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
        
        self._initialize_board()

    def _initialize_board(self):
        """Helper function to set up pieces"""
        # Reset board to zeros
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
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        # Set active agents
        self.agents = self.possible_agents[:]
        
        # Re-initialize board with pieces
        self._initialize_board()
        
        # Initialize rewards, terminations, truncations, infos
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Agent selector for turn management
        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: tuple ((from_row, from_col), (to_row, to_col))
        """
        # Get current agent and player
        current_agent = self.agent_selection
        current_player = 0 if current_agent == "player_0" else 1
        
        # Execute the move
        from_pos, to_pos = action
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Get the piece
        piece = self.board[from_row, from_col]
        
        # Check if it's a jump (capture)
        row_diff = abs(to_row - from_row)
        if row_diff == 2:
            # It's a jump - remove captured piece
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            self.board[mid_row, mid_col] = 0  # Remove captured piece
        
        # Move the piece
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0
        
        # Check for king promotion
        if current_player == 0 and to_row == 5 and piece == 1:
            self.board[to_row, to_col] = 2  # Promote to king
        elif current_player == 1 and to_row == 0 and piece == -1:
            self.board[to_row, to_col] = -2  # Promote to king
        
        # Switch to next agent
        self.agent_selection = self._agent_selector.next()
        
        # Check if game is over
        self._check_game_over()


    def _check_game_over(self):
        """Check if current agent has any valid moves"""
        current_player = 0 if self.agent_selection == "player_0" else 1
        player_pieces = self.get_player_pieces(current_player)
        
        has_moves = False
        for piece_pos in player_pieces:
            if len(self.get_moves_for_piece(piece_pos, current_player)) > 0:
                has_moves = True
                break
        
        if not has_moves:
            # Current player has no moves - they lose
            opponent = 1 - current_player
            self.rewards[f"player_{opponent}"] = 1
            self.rewards[f"player_{current_player}"] = -1
            self.terminations = {agent: True for agent in self.agents}
        else:
            # Game continues - no rewards yet
            self.rewards = {agent: 0 for agent in self.agents}
    
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

    
    def get_moves_for_piece(self, pos, player):
        """
        Get all valid moves for a piece at a given position
        
        Args:
            pos: tuple (row, col) - piece position
            player: 0 or 1 - which player owns the piece
        
        Returns:
            list of tuples: [(to_row, to_col), ...] - all valid destination positions
        """
        row, col = pos
        piece = self.board[row, col]
        
        # Check if piece belongs to player (can we removed; check function get_player_pieces )
        if player == 0 and piece <= 0:
            return []
        if player == 1 and piece >= 0:
            return []
        
        is_king = abs(piece) == 2
        valid_moves = []
        
        # Define possible directions
        if is_king:
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            if player == 0:
                directions = [(1, 1), (1, -1)]  # Move up
            else:
                directions = [(-1, 1), (-1, -1)]  # Move down
        
        # Check each direction
        for dr, dc in directions:
            # Simple move (1 square)
            to_row = row + dr
            to_col = col + dc
            
            if 0 <= to_row < 6 and 0 <= to_col < 6:
                if self.board[to_row, to_col] == 0:  # Empty square
                    valid_moves.append((to_row, to_col))
            
            # Jump move (2 squares)
            jump_row = row + 2 * dr
            jump_col = col + 2 * dc
            mid_row = row + dr
            mid_col = col + dc
            
            if 0 <= jump_row < 6 and 0 <= jump_col < 6:
                mid_piece = self.board[mid_row, mid_col]
                landing = self.board[jump_row, jump_col]
                
                # Valid jump: opponent in middle, empty landing
                if landing == 0:
                    if player == 0 and mid_piece < 0:  # Capture player 1's piece
                        valid_moves.append((jump_row, jump_col))
                    elif player == 1 and mid_piece > 0:  # Capture player 0's piece
                        valid_moves.append((jump_row, jump_col))
        
        return valid_moves

    def get_player_pieces(self, player):
        """
        Get positions of all pieces belonging to a player
        
        Args:
            player: 0 or 1 - which player
        
        Returns:
            list of tuples: [(row, col), ...] - positions of all player's pieces
        """
        pieces = []
        
        for row in range(6):
            for col in range(6):
                piece = self.board[row, col]
                
                # Player 0 has positive values (1 or 2)
                if player == 0 and piece > 0:
                    pieces.append((row, col))
                
                # Player 1 has negative values (-1 or -2)
                elif player == 1 and piece < 0:
                    pieces.append((row, col))
        
        return pieces


# Example usage:
# player_0_pieces = env.get_player_pieces(0)
# print(f"Player 0 pieces at: {player_0_pieces}")
# Output: [(0, 1), (0, 3), (0, 5), (1, 0), (1, 2), (1, 4)]

# Then get all moves:
# all_moves = []
# for piece_pos in player_0_pieces:
#     moves = env.get_moves_for_piece(piece_pos, player=0)
#     for move in moves:
#         all_moves.append((piece_pos, move))
# print(f"All possible moves: {all_moves}")





# Usage example
if __name__ == "__main__":
    env = Checkers6x6(render_mode="human")
    env.reset()
    env.render()
    