# Test script - add this at the bottom of your file or in a separate test file

from mycheckersenv import Checkers6x6

if __name__ == "__main__":
    print("=" * 60)
    print("Testing 6x6 Checkers Environment")
    print("=" * 60)

    # Create environment
    env = Checkers6x6(render_mode="human")
    env.reset()

    print("\n1. Initial Board State:")
    env.render()

    print("\n2. Player 0's pieces:")
    player_0_pieces = env.get_player_pieces(0)
    print(f"Positions: {player_0_pieces}")

    print("\n3. Valid moves for piece at (1, 0):")
    moves = env.get_moves_for_piece((1, 0), player=0)
    print(f"Can move to: {moves}")

    print("\n4. Making a move: (1, 0) -> (2, 1)")
    env.step(env.encode_action((1, 0), (2, 1)))
    env.render()
    print(f"Current player: {env.agent_selection}")
    print(f"Terminations: {env.terminations}")
    print(f"Rewards: {env.rewards}")

    print("\n5. Player 1's turn - piece at (4, 1):")
    moves = env.get_moves_for_piece((4, 1), player=1)
    print(f"Can move to: {moves}")

    print("\n6. Making a move: (4, 1) -> (3, 0)")
    env.step(env.encode_action((4, 1), (3, 0)))
    env.render()
    print(f"Current player: {env.agent_selection}")

    print("\n7. Testing a jump - set up a capture scenario:")
    # Manually set up a jump situation for testing
    env.board[2, 1] = 0  # Remove player 0 piece
    env.board[3, 2] = 1  # Put player 0 piece here
    env.board[4, 3] = -1  # Put player 1 piece here (to be captured)
    env.board[5, 4] = 0  # remove player 1 piece for jump capture
    env.render()

    print("\n8. Player 0 can jump from (3, 2) to (5, 4):")
    moves = env.get_moves_for_piece((3, 2), player=0)
    print(f"Available moves: {moves}")

    if (5, 4) in moves:
        print("\n9. Executing jump move:")
        env.step(env.encode_action((3, 2), (5, 4)))
        env.render()
        print(f"Piece at (4, 3) should be captured: {env.board[4, 3]}")
        print(f"Piece at (5, 4) should be promoted to king: {env.board[5, 4]}")

        print("\n9.1 Test King moves : ")
        print(f"Available moves: {env.get_moves_for_piece((5, 4), player=0)}")

    print("\n10. Test game over - remove all Player 1 pieces:")
    env.reset()
    # Remove all player 1 pieces
    for row in range(4, 6):
        for col in range(6):
            if env.board[row, col] < 0:
                env.board[row, col] = 0

    env.render()
    print("Making a move to trigger game over check...")
    env.step(env.encode_action((1, 0), (2, 1)))

    print(f"\nGame Over: {env.terminations}")
    print(f"Final Rewards: {env.rewards}")

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
