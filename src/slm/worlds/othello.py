"""Othello as a fully observable, verifiable world for world-model experiments (EXP-0038).

Board squares are indexed 0..63 row-major (a1 = 0, h8 = 63). A game is a sequence of moves; the
standard opening position is fixed, black moves first, and a player without a legal move passes
(passes are implicit in the sequence, as in Li et al., 2023, "Emergent World Representations").

Token vocabulary for sequence models: 0 = end-of-game, 1..64 = square + 1. The board state after each
move is returned from the mover-relative viewpoint used by Nanda et al. (2023): 0 = empty,
1 = belongs to the player who just moved, 2 = belongs to the opponent.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
EOS = 0
VOCAB_SIZE = 65


def initial_board() -> np.ndarray:
    board = np.zeros(64, dtype=np.int8)
    board[27] = board[36] = WHITE  # d4, e5
    board[28] = board[35] = BLACK  # e4, d5
    return board


def flips(board: np.ndarray, square: int, player: int) -> list[int]:
    """Squares flipped if ``player`` plays ``square``; empty list if the move is illegal."""
    if board[square] != EMPTY:
        return []
    opponent = 3 - player
    row, col = divmod(square, 8)
    flipped: list[int] = []
    for dr, dc in DIRECTIONS:
        r, c, line = row + dr, col + dc, []
        while 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == opponent:
            line.append(r * 8 + c)
            r, c = r + dr, c + dc
        if line and 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == player:
            flipped.extend(line)
    return flipped


def legal_moves(board: np.ndarray, player: int) -> list[int]:
    return [s for s in range(64) if board[s] == EMPTY and flips(board, s, player)]


def apply_move(board: np.ndarray, square: int, player: int) -> np.ndarray:
    flipped = flips(board, square, player)
    if not flipped:
        raise ValueError(f"illegal move {square} for player {player}")
    new = board.copy()
    new[square] = player
    new[flipped] = player
    return new


class Game:
    """Replays a move sequence, tracking the player to move and handling implicit passes."""

    def __init__(self) -> None:
        self.board = initial_board()
        self.player = BLACK

    def legal(self) -> list[int]:
        moves = legal_moves(self.board, self.player)
        if not moves and legal_moves(self.board, 3 - self.player):
            self.player = 3 - self.player  # pass
            moves = legal_moves(self.board, self.player)
        return moves

    def play(self, square: int) -> int:
        """Play ``square`` for the player to move (after any pass); return that player."""
        moves = self.legal()
        if square not in moves:
            raise ValueError(f"illegal move {square}")
        mover = self.player
        self.board = apply_move(self.board, square, mover)
        self.player = 3 - mover
        return mover


def random_game(rng: np.random.Generator, policy: str = "uniform") -> list[int]:
    """Moves of one complete game. ``uniform``: uniformly random legal move (synthetic dataset of
    Li et al.); ``greedy``: the move flipping the most discs, ties broken at random (a different
    play style, used as a far-transfer split)."""
    game, moves = Game(), []
    while True:
        legal = game.legal()
        if not legal:
            return moves
        if policy == "uniform":
            square = int(rng.choice(legal))
        elif policy == "greedy":
            gains = np.array([len(flips(game.board, s, game.player)) for s in legal])
            best = np.flatnonzero(gains == gains.max())
            square = legal[int(rng.choice(best))]
        else:
            raise ValueError(f"unknown policy {policy!r}")
        game.play(square)
        moves.append(square)


def states(moves: list[int]) -> np.ndarray:
    """Mover-relative board after each move: array (len(moves), 64) with 0 empty, 1 mine, 2 theirs."""
    game, out = Game(), np.zeros((len(moves), 64), dtype=np.int8)
    for t, square in enumerate(moves):
        mover = game.play(square)
        board = game.board
        out[t] = np.where(board == EMPTY, 0, np.where(board == mover, 1, 2))
    return out


def legal_after(moves: list[int]) -> Iterator[list[int]]:
    """Legal next moves after each prefix (empty when the game is over)."""
    game = Game()
    for square in moves:
        game.play(square)
        yield game.legal()


def encode(moves: list[int]) -> list[int]:
    return [m + 1 for m in moves] + [EOS]
