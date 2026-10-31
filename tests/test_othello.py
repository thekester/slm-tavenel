import numpy as np
import pytest

from slm.worlds import othello


def test_opening_moves_and_first_flip():
    board = othello.initial_board()
    assert sorted(othello.legal_moves(board, othello.BLACK)) == [19, 26, 37, 44]  # d3 c4 f5 e6
    after = othello.apply_move(board, 19, othello.BLACK)
    assert after[19] == after[27] == othello.BLACK  # d3 placed, d4 flipped
    assert (after != othello.EMPTY).sum() == 5


def test_illegal_move_rejected():
    game = othello.Game()
    with pytest.raises(ValueError):
        game.play(0)


@pytest.mark.parametrize("policy", ["uniform", "greedy"])
def test_random_games_are_legal_and_terminal(policy):
    rng = np.random.default_rng(0)
    for _ in range(20):
        moves = othello.random_game(rng, policy)
        game = othello.Game()
        for square in moves:
            game.play(square)
        assert game.legal() == []  # nobody can move at the end
        assert len(set(moves)) == len(moves) <= 60


def test_mover_relative_states_match_replay():
    rng = np.random.default_rng(1)
    moves = othello.random_game(rng)
    s = othello.states(moves)
    assert s.shape == (len(moves), 64)
    for t, square in enumerate(moves):
        assert s[t, square] == 1  # the square just played belongs to the mover
        assert (s[t] != 0).sum() == 4 + t + 1
    assert othello.encode(moves)[-1] == othello.EOS
    assert max(othello.encode(moves)) <= 64
