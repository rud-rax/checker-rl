import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def env(render_mode=None):
    """Factory function with wrappers"""
    env = raw_env(render_mode=render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "checkers_6x6_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(2))))
        self.render_mode = render_mode

        # Initialize board
        self.board = np.zeros((6, 6), dtype=np.int8)
        self._initialize_board()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "observation": spaces.Box(low=-2, high=2, shape=(6, 6), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(1296,), dtype=np.int8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(1296)

    def observe(self, agent):
        """Return observation for given agent"""
        player = 0 if agent == "player_0" else 1
        return {
            "observation": self.board.copy(),
            "action_mask": self.get_action_mask(player),
        }

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Reset board
        self._initialize_board()

        # Agent selector
        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.step_count = 0
        self.max_steps = 500

    def step(self, action):
        """Execute one step"""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # Increment step counter
        self.step_count += 1

        agent = self.agent_selection

        # Reset cumulative rewards for THIS agent at start of their turn
        self._cumulative_rewards[agent] = 0

        # Decode and execute action
        from_pos, to_pos = self.decode_action(action)
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        piece = self.board[from_row, from_col]

        # Track if piece was captured (for reward shaping)
        piece_captured = False

        # Handle jump
        row_diff = abs(to_row - from_row)
        if row_diff == 2:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            self.board[mid_row, mid_col] = 0
            piece_captured = True

        # Move piece
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0

        # King promotion
        current_player = 0 if agent == "player_0" else 1
        if current_player == 0 and to_row == 5 and piece == 1:
            self.board[to_row, to_col] = 2
        elif current_player == 1 and to_row == 0 and piece == -1:
            self.board[to_row, to_col] = -2

        # Calculate shaped reward for this move
        shaped_reward = self.calculate_reward(current_player, action, piece_captured)

        # Check for game over BEFORE switching agents
        next_player = 1 - current_player
        next_player_pieces = self.get_player_pieces(next_player)
        has_moves = False
        for piece_pos in next_player_pieces:
            if len(self.get_moves_for_piece(piece_pos, next_player)) > 0:
                has_moves = True
                break

        # Check for game termination or truncation
        if not has_moves:
            # Next player has no moves - current player wins
            self.rewards[agent] = 1.0 + shaped_reward
            self.rewards[self.agents[1 - self.agent_name_mapping[agent]]] = -1.0
            self.terminations = {a: True for a in self.agents}
        elif self.step_count >= self.max_steps:
            # Game too long - truncate with draw
            self.rewards[agent] = shaped_reward  # Keep shaped reward
            self.rewards[self.agents[1 - self.agent_name_mapping[agent]]] = 0.0
            self.truncations = {a: True for a in self.agents}
        else:
            # Game continues - give shaped reward
            self.rewards[agent] = shaped_reward
            self._clear_rewards()  # Clear other agent's reward

        # Select next agent
        self.agent_selection = self._agent_selector.next()

        # Accumulate rewards AFTER setting them
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode == "human":
            self.print_board(self.board)

    def print_board(self, board):
        symbols = {0: "·", 1: "○", 2: "⊙", -1: "●", -2: "⊗"}
        print("\n  a b c d e f")
        print("  ───────────")
        for row in range(5, -1, -1):
            row_str = f"{row + 1}│"
            for col in range(6):
                row_str += symbols[board[row, col]] + " "
            print(row_str)
        print()

    # Helper methods (keep all your existing helper methods)
    def _initialize_board(self):
        self.board = np.zeros((6, 6), dtype=np.int8)
        for row in range(2):
            for col in range(6):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1
        for row in range(4, 6):
            for col in range(6):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1

    def encode_action(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        return from_row * 216 + from_col * 36 + to_row * 6 + to_col

    def decode_action(self, action):
        from_row = action // 216
        action = action % 216
        from_col = action // 36
        action = action % 36
        to_row = action // 6
        to_col = action % 6
        return ((from_row, from_col), (to_row, to_col))

    def get_action_mask(self, player):
        mask = np.zeros(1296, dtype=np.int8)
        player_pieces = self.get_player_pieces(player)
        for piece_pos in player_pieces:
            valid_moves = self.get_moves_for_piece(piece_pos, player)
            for to_pos in valid_moves:
                action = self.encode_action(piece_pos, to_pos)
                mask[action] = 1
        return mask

    def get_player_pieces(self, player):
        pieces = []
        for row in range(6):
            for col in range(6):
                piece = self.board[row, col]
                if player == 0 and piece > 0:
                    pieces.append((row, col))
                elif player == 1 and piece < 0:
                    pieces.append((row, col))
        return pieces

    def get_moves_for_piece(self, pos, player):
        row, col = pos
        piece = self.board[row, col]
        if player == 0 and piece <= 0:
            return []
        if player == 1 and piece >= 0:
            return []

        is_king = abs(piece) == 2
        simple_moves = []
        jump_moves = []

        if is_king:
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            if player == 0:
                directions = [(1, 1), (1, -1)]
            else:
                directions = [(-1, 1), (-1, -1)]

        for dr, dc in directions:
            # Simple move
            to_row = row + dr
            to_col = col + dc
            if 0 <= to_row < 6 and 0 <= to_col < 6:
                if self.board[to_row, to_col] == 0:
                    simple_moves.append((to_row, to_col))

            # Jump move
            jump_row = row + 2 * dr
            jump_col = col + 2 * dc
            mid_row = row + dr
            mid_col = col + dc
            if 0 <= jump_row < 6 and 0 <= jump_col < 6:
                mid_piece = self.board[mid_row, mid_col]
                landing = self.board[jump_row, jump_col]
                if landing == 0:
                    if player == 0 and mid_piece < 0:
                        jump_moves.append((jump_row, jump_col))
                    elif player == 1 and mid_piece > 0:
                        jump_moves.append((jump_row, jump_col))

        return jump_moves if len(jump_moves) > 0 else simple_moves

    def calculate_reward(self, current_player, action, piece_captured):
        """
        Calculate reward for a move (Simple Shaping)

        Args:
            current_player: 0 or 1
            action: Action taken
            piece_captured: Whether a piece was captured

        Returns:
            reward: Immediate reward for this action
        """
        reward = 0.0

        # Reward for capturing opponent piece
        if piece_captured:
            reward += 0.5

        # Reward for king promotion
        from_pos, to_pos = self.decode_action(action)
        to_row, to_col = to_pos
        piece = self.board[to_row, to_col]

        if abs(piece) == 2:  # Is a king
            reward += 0.3

        # Small penalty for each move (encourage winning quickly)
        reward -= 0.01

        return reward
