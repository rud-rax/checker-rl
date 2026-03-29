# Environment Info

### 🎯 What skill should the agent learn?

- Navigate through a maze? No
- Balance and control a system? No
- Optimize resource allocation? No
- Play a strategic game? Yes

### 👀 What information does the agent need? 

- Position and velocity? No
- Current state of the system? YES
- Historical data? No
- Partial or full observability? Full observability

### 🎮 What actions can the agent take?

- Discrete choices (move up/down/left/right)? 
left diagonal up, right diagonal up, left diagonal down, right diagonal down
- Continuous control (steering angle, throttle)? no
- Multiple simultaneous actions? no

### 🏆 How do we measure success?

- Reaching a specific goal? no
- Minimizing time or energy? no
- Maximizing a score? yes
- Avoiding failures? yes ?!

### ⏰ When should episodes end?

- Task completion (success/failure)? yes, both
- Time limits? no
- Safety constraints? no