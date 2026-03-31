# Implementation Checklist
## Phase 1: Core Mechanics

- [x] Initialize 6x6 board correctly
- [x] Implement diagonal move validation
- [x] Implement jump/capture logic
- [x] Handle mandatory jumping rule
- [x] King promotion logic
- [x] Turn switching

## Phase 2: AEC Integration

- [x] Define observation/action spaces
- [x] Implement reset()
- [x] Implement step(action)
- [x] Implement observe(agent)
- [x] Handle action masking
- [x] Implement termination detection

## Phase 3: Multi-Jump & Edge Cases

- [ ] Multi-jump sequences
- [ ] King backward movement
- [ ] No valid moves detection
- [ ] Draw conditions

## Phase 4: Rendering

- [ ] Text-based board display
- [ ] Pygame/matplotlib visualization (optional)

## Implementation Checklist

- [ ]  Build Actor network (36 → 128 → 128 → 1296)
- [ ]  Build Critic network (36 → 128 → 64 → 1)
- [ ]  Implement action selection with masking
- [ ]  Implement advantage calculation (TD error)
- [ ]  Implement actor update (policy gradient)
- [ ]  Implement critic update (value function)
- [ ]  Training loop with self-play
- [ ]  Save/load checkpoints
- [ ]  Evaluation/testing code