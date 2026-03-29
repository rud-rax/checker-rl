# Implementation Checklist
## Phase 1: Core Mechanics

- [ ] Initialize 6x6 board correctly
- [ ] Implement diagonal move validation
- [ ] Implement jump/capture logic
- [ ] Handle mandatory jumping rule
- [ ] King promotion logic
- [ ] Turn switching

## Phase 2: AEC Integration

- [ ] Define observation/action spaces
- [ ] Implement reset()
- [ ] Implement step(action)
- [ ] Implement observe(agent)
- [ ] Handle action masking
- [ ] Implement termination detection

## Phase 3: Multi-Jump & Edge Cases

- [ ] Multi-jump sequences
- [ ] King backward movement
- [ ] No valid moves detection
- [ ] Draw conditions

## Phase 4: Rendering

- [ ] Text-based board display
- [ ] Pygame/matplotlib visualization (optional)