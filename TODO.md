# Implementation Checklist
## Phase 1: Core Mechanics

- [x] Initialize 6x6 board correctly
- [x] Implement diagonal move validation
- [x] Implement jump/capture logic
- [x] Handle mandatory jumping rule
- [x] King promotion logic
- [ ] Turn switching

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