# turboquant
Standalone TurboQuant KV Cache Inference 
Example uses: https://huggingface.co/g023/Qwen3-1.77B-g023 for demonstration

~~~
run_tquant.py — Standalone TurboQuant KV Cache Inference for https://huggingface.co/g023/Qwen3-1.77B-g023
Author: g023 (https://github.com/g023) (https://huggingface.co/g023) 

Implements TurboQuant (ICLR 2026, arXiv:2504.19874) KV cache compression
directly inside a Transformers inference script. All algorithms are self-contained. Minimal dependencies. 

Algorithms:
  1. Random orthogonal rotation (QR decomposition) → Beta-distributed coordinates
  2. Lloyd-Max optimal scalar quantization → MSE-optimal centroids
  3. QJL sign-bit residual correction → unbiased inner products
  4. Group quantization for values → per-group min-max

Model Repo: [ https://huggingface.co/g023/Qwen3-1.77B-g023 ]
Model Info: (head_dim=128, 8 KV heads, 29 layers, GQA)
Model Instructions: Download files from MODEL REPO and throw in ./Qwen3-BEST and then run this program. 

Reqs:
pip install transformers datasets scipy
~~~

# instructions:
1) Download the model files and throw in ./Qwen3-BEST
2) Install pre-reqs: `pip install transformers datasets scipy`
3) Run `python run_tquant.py` for single-shot inference
4) Run `python run_tquant.py -i` or `python run_tquant.py --interactive` for multi-turn chat

# requirements:
- Python 3.10+
- transformers >= 5.0.0 (uses new layer-based Cache API)
- scipy
- CUDA GPU (tested on RTX 3060 12GB)

# features:
- TurboQuant KV cache compression (~1.55x memory savings)
- 4-bit key quantization (3-bit MSE + 1-bit QJL residual)
- 4-bit value quantization (group min-max)
- Mirostat V2 sampling (arXiv:2007.14966v2) for controlled perplexity and reduced repetition
- Qwen3 thinking mode support (automatic `<think>` tag parsing)
- Multi-turn interactive chat mode (`-i` / `--interactive`)
- Streaming output with real-time token display
- Self-test on startup validates quantization math
- Accurate token counting from model output
- Pre-computed Lloyd-Max codebooks (cached in ./codebooks/)

# generation parameters:
Custom sampling with Mirostat V2 (arXiv:2007.14966v2) for controlled perplexity:
- temperature: 0.7 (applied before Mirostat)
- mirostat_tau: 5.0 (target surprise, -ln p)
- mirostat_eta: 0.1 (learning rate for mu adaptation)
- repetition_penalty: 1.2 (multiplicative, windowed)
- frequency_penalty: 0.15 (additive, scales with token count in window)
- presence_penalty: 0.15 (additive, flat per unique token in window)
- rep_penalty_window: 256 (tokens to consider for penalties)

# results (RTX 3060, single-shot):
~~~
------------------------------------------------------------
Results:
  Total tokens: 1040
  Time: 24.79s
  Perplexity: 1.33
  Tokens/sec: 41.95

  TurboQuant Memory Report:
    Sequence length:    1087
    Compressed tokens:  771
    Buffer tokens:      316
    Compressed layers:  24
    Full prec. layers:  5
    Actual KV cache:    74.01 MB
    Full precision:     123.14 MB
    Compression ratio:  1.66x
    Savings:            49.13 MB
------------------------------------------------------------
~~~

# multi-turn example:
~~~
$ python run_tquant.py -i
TurboQuant Interactive Chat (type 'quit' or 'exit' to stop)

You: What is 2+2?
Assistant: 2 + 2 = 4.
  [110 tokens, 2.1s, 52.4 tok/s]

You: Multiply that by 3
Assistant: 4 × 3 = 12.
  [109 tokens, 1.8s, 60.6 tok/s]

You: quit
[Exiting]
~~~

# single run output:
~~~
============================================================
TurboQuant Self-Test
============================================================
  Key quantization (4-bit TurboQuantProd):
    Avg cosine similarity: 0.974612
    MSE: 5.298514e-02
    Packed MSE indices shape: torch.Size([1, 8, 256, 64])
    Packed QJL signs shape:   torch.Size([1, 8, 256, 16])
  Value quantization (4-bit group):
    Avg cosine similarity: 0.996977
    MSE: 6.131127e-03
    Packed data shape: torch.Size([1, 8, 256, 64])
  Key compression ratio: 3.05x
  Value compression ratio: 3.20x
  Inner product accuracy (TQ guarantee):
    Correlation:    0.974794
    Relative bias:  -0.000344

  Result: PASS ✓
============================================================
[Model] Loading model...
Loading weights: 100%|█| 321/321 [00:00<00:00, 675.79it/s, Materializing param=model.norm.weight
[Model] Loaded on cuda:0, dtype=torch.bfloat16

============================================================
Inference with TurboQuant KV Cache
============================================================

[TQ] Cache: 4-bit keys, 4-bit values, buffer=256, compressing 24/29 layers
<think>
Okay, let's tackle this problem. The user is working on an arcade game in JavaScript and needs help creating a plan or writing tests. They mentioned using the tools: rationalize, red_green_tdd, and create_plan.

First, I need to recall what each tool does. Rationalize probably helps break down problems into smaller parts. Red-green-tDD (Red-Green-Refactor) is a test-driven approach where you write tests first, then code, then refactor. Create_plan would be for outlining steps or features.

The user hasn't provided specific details about the game yet. But since they're starting with the next step, maybe they have some existing code or outline. However, without that, I should think of common arcade games as examples. Maybe something like pong, space invaders, or a simple platformer.

Assuming they want to structure their game logic. Let's say they need to handle player movement, collision detection, scoring, etc. 

Using Rethink:
1. Write tests before coding.
2. Use test cases to check functionality.
3. Refactor after testing.

For example, if the game has a player moving left/right, tests can verify that pressing arrow keys changes direction correctly.

Create_plan might involve breaking down features into tasks. Like designing the game loop, handling input, updating game state, rendering graphics, checking collisions.

But since there's no actual problem statement here, perhaps the assistant needs to prompt them for more info? Or assume a generic scenario?

Wait, but in the initial instructions, the user says "Synthesize their output when reasoning." So the assistant must combine the outputs from rationalize, red_green_tdd, and create_plan.

Maybe the assistant is supposed to generate a plan based on these tools. For instance:

- Rationalize: Break down the game into components (player, enemy, score).
- Red-green-TDD: Create tests for each component first.
- Create-plan: Outline steps needed to build each part.

So putting it all together, the answer would be a structured plan with test scenarios and code steps.
</think>

To develop an arcade game in JavaScript using **Rationalize**, **Red-Green-TDD** (Test-Driven Development), and **create_plan**, follow this structured approach:

---

### **1. Plan & Design**
#### **Use `create_plan`**:  
Break down the game into modular components:
- **Player Movement**: Handle keyboard inputs (arrow keys) and update position.
- **Game Loop**: Update game state (physics, collision detection) at fixed intervals.
- **Collision Detection**: Check for player/enemy collisions to handle scoring or game over conditions.
- **Rendering**: Use HTML5 Canvas or PIXI.js to draw the game loop.

```javascript
// Example: Game Loop Structure
function updateGameState() {
  // Player movement logic here
}

function renderGame() {
  // Draw player, enemies, etc. onto canvas
}
```

---

### **2. Test-Driven Development (Red-Green-TDD)**  
Write unit tests before coding:

#### **Example Tests (for Player Movement):**

| Scenario | Input | Expected Output |
|----------|--------|----------------|
| Move Left | Arrow key pressed | Player.x -= speed; |
| Move Right | Arrow key released | Player.x += speed; |

```js
describe("Player Movement", () => {
  it('should move left when arrow key is pressed', () => {
    const player = { x: 0 };
    fireEvent(player, 'keydown', { keyCode: ArrowLeft });
    expect(player.x).toBe(-speed);
  });

  it('should move right when arrow key is released', () => {
    const player = { x: 0 };
    fireEvent(player, 'keyup', {keyCode: ArrowRight});
    expect(player.x).toBe(0 + speed); 
  });
});
```

---

### **3. Rationalize**  
Break down complex tasks into smaller steps:
1. **Initialize game loop**: Set interval for updates.
2. **Handle input events**: Listen to keyboard events and update state.
3. **Physics logic**: Update position based on velocity/acceleration.
4. **Collision detection**: Check boundaries and detect collisions with enemies.

---

### **4. Integration Plan**
- **Step-by-step plan**:
  - Create `player.js` with movement functions.
  - Implement `gameLoop()` using requestAnimationFrame().
  - Add collision checks in `collisionDetection.js`.
  - Write tests for each function (e.g., `movePlayer`, `checkCollisions`).

```js
// Example code structure
const Player = {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  },
  moveLeft() {
    this.x -= speed; // Rationalize: break into separate methods
  }
};

function gameLoop() {
  updateGameState();
  renderGame();
}
```

---

### **5. Refactor & Optimize**  
After coding a feature like player movement, refactor the engine’s physics or use PIXI.js for rendering if performance is an issue.

---

By combining these tools, you ensure your game works as expected through testing before refactoring, leading to maintainable, scalable code.

------------------------------------------------------------
Results:
  Total tokens: 1071
  Time: 25.87s
  Perplexity: 1.27
  Tokens/sec: 41.40

  TurboQuant Memory Report:
    Sequence length:    1118
    Compressed tokens:  771
    Buffer tokens:      347
    Compressed layers:  24
    Full prec. layers:  5
    Actual KV cache:    77.52 MB
    Full precision:     126.65 MB
    Compression ratio:  1.63x
    Savings:            49.13 MB
------------------------------------------------------------
~~~
