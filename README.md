# Filipino Dama

This is a computer version of **Filipino Dama**, a traditional Filipino board game similar to checkers (draughts). You can play it on your computer with a friend, or against the computer itself.

The game comes with **two different computer opponents**: the **Calculating Opponent**, which thinks ahead before every move, and the **Learning Opponent**, an artificial intelligence that taught itself to play. More about them below.

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

This one is an artificial intelligence that taught itself how to play. Nobody programmed its strategy. Instead, it learned by playing thousands and thousands of games against itself, gradually noticing which moves lead to winning and which lead to losing. The more it trains, the stronger it gets.

At a glance:

| | Calculating Opponent | Learning Opponent |
|---|---|---|
| How it plays | Works out moves ahead, every turn | Uses what it learned from self-practice |
| Strength | Four fixed difficulty levels | Grows the more it is trained |
| Where it comes from | Built-in rules and logic | Trained by the included practice system |

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

## What is in this project folder?

- The game itself (board, rules, and the window you play in)
- The two computer opponents
- The training system that teaches the Learning Opponent
- Saved AI "brains" from earlier training, settings files, and logs

For developers: the source code lives in the `src` folder, the training settings in the `config` folder, and helper scripts at the project root and in `scripts`.
