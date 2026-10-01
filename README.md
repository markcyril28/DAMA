# Filipino Dama

This is a computer version of **Filipino Dama**, a traditional Filipino board game similar to checkers (draughts). You can play it on your computer with a friend, or against the computer itself.

At its heart, this project is a **combination of three things**:

- A **GUI game**: a game window where you point, click, and play
- A classic **algorithm**: precise, step-by-step calculation that thinks ahead
- **Reinforcement learning**: an AI that improves through trial and error, learning from its own wins and losses

The last two power the game's **two different computer opponents**: the **Calculating Opponent**, which thinks ahead before every move, and the **Learning Opponent**, an artificial intelligence that taught itself to play. More about them below.

## What is Filipino Dama?

Filipino Dama is played on the dark squares of a standard 8x8 board. Each player starts with 12 pieces and tries to capture all of the opponent's pieces.

The basic rules:

- Pieces move diagonally, one square at a time, toward the opponent's side
- You capture an opponent's piece by jumping over it
- If a capture is possible, you must take it
- If your piece can keep jumping after a capture, it keeps going in the same turn
- When a piece reaches the far end of the board, it becomes a **king** (called a "dama")
- Kings are powerful: they can slide and capture along a whole diagonal, at any distance

## The two computer opponents

### 1. The Calculating Opponent

This opponent thinks ahead several moves before choosing the best one, like a careful chess player working out "if I go here, they go there...". It comes in four difficulty levels, from easy to very hard. The harder the level, the more time it spends thinking.

### 2. The Learning Opponent

This one is an artificial intelligence that taught itself how to play. Nobody programmed its strategy. Instead, it learned by playing thousands and thousands of games against itself, gradually noticing which moves lead to winning and which lead to losing. This trial-and-error style of teaching is known as **reinforcement learning**. The more it trains, the stronger it gets.

At a glance:

| | Calculating Opponent | Learning Opponent |
|---|---|---|
| How it plays | Works out moves ahead, every turn | Uses what it learned from self-practice |
| Strength | Four fixed difficulty levels | Grows the more it is trained |
| Where it comes from | Built-in rules and logic (a classic algorithm) | Trained by the included practice system (reinforcement learning) |

## How does the Learning Opponent improve?

The project includes a training system that works like a practice loop:

1. The AI plays many games against itself (and against the Calculating Opponent)
2. It studies those games and adjusts how it judges moves
3. It is tested against the Calculating Opponent to measure progress
4. Its progress is saved, so training can continue later from where it left off

This runs on a computer with a powerful graphics card, which does the heavy number-crunching. Training can run for hours or days; the longer it runs, the better the AI tends to play.

## How do I start the game?

The game is built with the Python programming language. If Python and the game's requirements are set up on your computer, you start it by running this from the project folder:

```
bash run_game.sh
```

A window opens with the board. You click a piece to select it, then click the square you want to move it to. From the menus you can choose who plays: you, a friend, the Calculating Opponent, or the Learning Opponent, and adjust settings like difficulty and appearance.

## Running on a Mac (Apple Silicon)

The project also runs on Macs with an Apple chip (M1 or newer). A Mac has no NVIDIA graphics card, so training uses the Mac's own graphics chip through Apple's Metal system (called "MPS") instead. Each main script has a Mac version:

```
bash setup_conda_mac.sh              # once: creates the 'dama' Python environment and builds the speed-ups
bash run_game_mac.sh                 # play the game
bash local_train_mac.sh              # train the Learning Opponent (settings: config/training_config_mac.yaml)
bash stop_training_mac.sh            # stop training
bash eval_checkpoints_mac.sh --once  # test the saved models against the Calculating Opponent
```

Before the first setup you need Miniforge or Miniconda for Apple Silicon (`brew install --cask miniforge`) and Apple's command line tools (`xcode-select --install`), which compile the speed-up modules. The Mac scripts switch to the Python environment by themselves.

What is different on a Mac:

- **Graphics chip:** training runs on the Mac's GPU in a compact number format (bfloat16), which needs macOS 14 or newer; older versions automatically fall back to a similar format (float16). Playing the game and the practice games run on the main processor: the model is small, and for one move at a time that is faster on a Mac than the graphics chip.
- **Shared memory:** a Mac's processor and graphics chip share one pool of memory. The Mac settings are sized for a 36 GB Mac: training uses a little over 20 GB of it (about 13 GB of that for the graphics chip), so close other memory-hungry apps while it runs.
- **Separate files:** Mac training saves to `models/checkpoints_mac/`, `models/latest_mac.pt` and `logs/mac/`, so it never mixes with training copied from another computer. To play against it, pick `models/latest_mac.pt` in the game's settings. The model has the same shape as on the other computers, so saved models can be copied between them.
- **Practice-game workers:** macOS starts them as fresh processes ("spawn") instead of copies of the trainer ("fork"), because copying a process after Apple's graphics libraries have loaded is unsafe. The `DAMA_MP_START_METHOD` setting overrides this when tracking down a problem. These workers exit by themselves when the trainer stops, even if it was force-quit; `stop_training_mac.sh` also ends any it finds left behind.
- **No overheating pause:** macOS does not let programs read its temperature sensors, so that safety pause is off. The Mac slows itself down when it runs hot.
- **Rosetta:** if Terminal runs under Rosetta (Apple's Intel translator), the Mac scripts restart themselves in native mode.

`eval_checkpoints_mac.sh` tests the models saved in `models/checkpoints_mac/` and writes the scores to `models/eval_results_mac.jsonl`, with a chart in `models/eval_progress_mac.png`. Without `--once` it keeps watching for new ones. The scripts without `_mac` in their names are for the Linux and Windows computers; on a Mac, use the `_mac` versions.
