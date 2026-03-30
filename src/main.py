# Test script - add this at the bottom of your file or in a separate test file
import numpy as np
from mycheckersenv import Checkers6x6

# Test script - add this at the bottom or in a separate test file

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE TEST: 6x6 Checkers with Action Encoding & Masking")
    print("=" * 70)
    
    # Create environment
    env = Checkers6x6(render_mode="human")
    env.reset()
    
    # TEST 1: Action Encoding/Decoding
    print("\n[TEST 1] Action Encoding/Decoding")
    print("-" * 70)
    from_pos = (1, 0)
    to_pos = (2, 1)
    encoded = env.encode_action(from_pos, to_pos)
    decoded = env.decode_action(encoded)
    print(f"Original move: {from_pos} -> {to_pos}")
    print(f"Encoded action: {encoded}")
    print(f"Decoded move: {decoded}")
    print(f"✓ Encoding/Decoding works: {decoded == (from_pos, to_pos)}")
    
    # TEST 2: Action Masking
    print("\n[TEST 2] Action Masking")
    print("-" * 70)
    mask = env.get_action_mask(player=0)
    print(f"Mask shape: {mask.shape}")
    print(f"Total actions: {len(mask)}")
    print(f"Valid actions: {np.sum(mask)}")
    print(f"Invalid actions: {len(mask) - np.sum(mask)}")
    
    # Show first 5 valid actions
    valid_action_indices = np.where(mask == 1)[0]
    print(f"\nFirst 5 valid action numbers: {valid_action_indices[:5]}")
    print("Decoded moves:")
    for action_num in valid_action_indices[:5]:
        move = env.decode_action(action_num)
        print(f"  Action {action_num}: {move}")
    
    # TEST 3: Observation Space
    print("\n[TEST 3] Observation")
    print("-" * 70)
    obs = env.observe("player_0")
    print(f"Observation keys: {obs.keys()}")
    print(f"Board shape: {obs['observation'].shape}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    print(f"Valid actions in observation: {np.sum(obs['action_mask'])}")
    
    # TEST 4: Play with Encoded Actions
    print("\n[TEST 4] Playing with Encoded Actions")
    print("-" * 70)
    env.reset()
    env.render()
    
    # Player 0 move
    print("\nPlayer 0's turn:")
    mask = env.get_action_mask(0)
    valid_actions = np.where(mask == 1)[0]
    action = valid_actions[0]  # Pick first valid action
    move = env.decode_action(action)
    print(f"Choosing action {action}: {move[0]} -> {move[1]}")
    env.step(action)
    env.render()
    
    # Player 1 move
    print("\nPlayer 1's turn:")
    mask = env.get_action_mask(1)
    valid_actions = np.where(mask == 1)[0]
    action = valid_actions[0]
    move = env.decode_action(action)
    print(f"Choosing action {action}: {move[0]} -> {move[1]}")
    env.step(action)
    env.render()
    
    # TEST 5: Test Jump with Encoded Actions
    print("\n[TEST 5] Testing Jump/Capture with Encoded Actions")
    print("-" * 70)
    env.reset()
    # Set up jump scenario
    env.board[5, 4] = 0
    env.board[3, 2] = 1  # Player 0 piece
    env.board[4, 3] = -1  # Player 1 piece (to be captured)
    env.render()
    
    print("\nLooking for jump move from (3,2) to (5,4):")
    jump_action = env.encode_action((3, 2), (5, 4))
    print(f"Jump action encoded as: {jump_action}")
    
    mask = env.get_action_mask(0)
    if mask[jump_action] == 1:
        print("✓ Jump action is in the mask!")
        print("\nExecuting jump...")
        env.step(jump_action)
        env.render()
        print(f"Captured piece removed: {env.board[4, 3] == 0}")
        print(f"Piece promoted to king: {env.board[5, 4] == 2}")
    else:
        print("✗ Jump action not found in mask")
    
    # TEST 6: Full Game with Random Actions
    print("\n[TEST 6] Playing Full Random Game")
    print("-" * 70)
    env.reset()
    
    move_count = 0
    max_moves = 50
    
    while not env.terminations["player_0"] and move_count < max_moves:
        current_player = 0 if env.agent_selection == "player_0" else 1
        
        # Get action mask and sample valid action
        mask = env.get_action_mask(current_player)
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            print(f"No valid actions for player {current_player}")
            break
        
        # Choose random valid action
        action = np.random.choice(valid_actions)
        move = env.decode_action(action)
        
        print(f"Move {move_count + 1}: Player {current_player} - {move[0]} -> {move[1]}")
        env.step(action)
        move_count += 1
    
    print(f"\nGame finished after {move_count} moves")
    print(f"Terminations: {env.terminations}")
    print(f"Rewards: {env.rewards}")
    env.render()
    
    # TEST 7: Verify Mandatory Jump Rule
    print("\n[TEST 7] Verify Mandatory Jump Rule")
    print("-" * 70)
    env.reset()
    # Set up scenario where jump is available
    env.board = np.zeros((6, 6), dtype=np.int8)
    env.board[2, 2] = 1  # Player 0 piece
    env.board[3, 3] = -1  # Player 1 piece (can be captured)
    env.board[2, 4] = 1  # Another player 0 piece (can make simple move)
    env.render()
    
    mask = env.get_action_mask(0)
    valid_actions = np.where(mask == 1)[0]
    print(f"Valid actions: {len(valid_actions)}")
    
    # Check if only jump moves are available
    has_jump = False
    has_simple = False
    for action in valid_actions:
        move = env.decode_action(action)
        from_pos, to_pos = move
        distance = abs(to_pos[0] - from_pos[0])
        if distance == 2:
            has_jump = True
            print(f"Jump found: {move}")
        elif distance == 1:
            has_simple = True
            print(f"Simple move found: {move}")
    
    print(f"\n✓ Mandatory jump enforced: {has_jump and not has_simple}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)