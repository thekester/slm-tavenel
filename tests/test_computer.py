import random

import pytest
import torch

from slm.computer import CompiledCalculator, DeltaRAM, LookupMLP, bits_to_byte, byte_to_bits, gated_delta_scan


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ram_matches_a_reference_memory_under_random_traffic(dtype):
    ram, reference = DeltaRAM(heads=4, rows=16, dtype=dtype), {}
    rng = random.Random(0)
    for _ in range(2000):
        address = rng.randrange(ram.size)
        if rng.random() < 0.5:
            value = rng.randrange(256)
            ram.write(address, value)
            reference[address] = value
        else:
            assert ram.read(address) == reference.get(address, 0)
    assert all(ram.read(a) == v for a, v in reference.items())
    # Exactness: every state entry is still exactly 0 or 1 after thousands of updates.
    assert set(ram.state.unique().tolist()) <= {0.0, 1.0}


def test_low_address_bits_pick_the_head():
    ram = DeltaRAM(heads=4, rows=8, dtype=torch.float32)
    ram.write(6, 0xAB)  # head 6 % 4 = 2, row 6 // 4 = 1
    assert ram.state[0, 2, :, 1].tolist() == [float(b) for b in byte_to_bits(0xAB)]
    assert ram.state[0, [0, 1, 3]].abs().sum() == 0


def test_clear_head_only_erases_that_head():
    ram = DeltaRAM(heads=2, rows=4)
    for address in range(ram.size):
        ram.write(address, address + 1)
    ram.clear_head(0)
    assert [ram.read(a) for a in range(ram.size)] == [0, 2, 0, 4, 0, 6, 0, 8]


def test_halving_a_head_makes_a_counter():
    ram = DeltaRAM(heads=1, rows=1, dtype=torch.float32)
    ram.write(0, 1)
    ram.halve_head(0)
    ram.halve_head(0)
    assert ram.state[0, 0, 0, 0].item() == 0.25


def test_overwrite_replaces_the_whole_byte():
    ram = DeltaRAM(heads=1, rows=2)
    ram.write(1, 0xFF)
    ram.write(1, 0x01)
    assert ram.read(1) == 0x01
    assert ram.read(0) == 0


def test_scan_is_the_textbook_recurrence():
    torch.manual_seed(0)
    b, h, t, dk, dv = 2, 3, 5, 4, 6
    q, k = torch.randn(b, h, t, dk), torch.nn.functional.normalize(torch.randn(b, h, t, dk), dim=-1)
    v, alpha, beta = torch.randn(b, h, t, dv), torch.rand(b, h, t), torch.rand(b, h, t)
    out, state = gated_delta_scan(q, k, v, alpha, beta)
    s = torch.zeros(b, h, dv, dk)
    eye = torch.eye(dk)
    for i in range(t):
        kk = k[:, :, i]
        s = alpha[:, :, i, None, None] * s @ (eye - beta[:, :, i, None, None] * kk[..., :, None] * kk[..., None, :]) \
            + beta[:, :, i, None, None] * v[:, :, i, :, None] * kk[..., None, :]
        torch.testing.assert_close(out[:, :, i], (s @ q[:, :, i, :, None])[..., 0])
    torch.testing.assert_close(state, s)


def test_bit_helpers_round_trip_and_reject_bad_input():
    assert all(bits_to_byte(torch.tensor(byte_to_bits(x))) == x for x in range(256))
    with pytest.raises(ValueError):
        byte_to_bits(256)
    with pytest.raises(ValueError):
        bits_to_byte(torch.tensor([2.0] + [0.0] * 7))
    with pytest.raises(IndexError):
        DeltaRAM(heads=2, rows=2).read(4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lookup_mlp_is_exact_on_every_entry(dtype):
    mlp = LookupMLP([10, 10, 2], [10, 2], lambda a, b, c: ((a + b + c) % 10, (a + b + c) // 10), dtype)
    for a in range(10):
        for b in range(10):
            for c in range(2):
                assert mlp(a, b, c) == ((a + b + c) % 10, (a + b + c) // 10)


def test_lookup_mlp_rejects_out_of_range_tables():
    with pytest.raises(ValueError):
        LookupMLP([3], [2], lambda a: a)


def test_compiled_calculator_tables_cover_every_digit_case():
    calc = CompiledCalculator(max_digits=4)
    for a in range(10):
        for b in range(10):
            assert calc.mul(a, b) == ((a * b) % 10, (a * b) // 10)
            for c in range(10):
                assert calc.acc(a, b, c) == ((a + b + c) % 10, (a + b + c) // 10)


def test_compiled_addition_on_a_two_digit_grid_and_random_large_operands():
    calc = CompiledCalculator(max_digits=12)
    for a in range(100):
        for b in range(0, 100, 7):
            assert calc.add_numbers(a, b) == a + b
    rng = random.Random(1)
    for _ in range(50):
        a, b = rng.randrange(10**12), rng.randrange(10**12)
        assert calc.add_numbers(a, b) == a + b


def test_compiled_multiplication_matches_python():
    calc = CompiledCalculator(max_digits=8)
    for a in range(0, 100, 3):
        for b in range(0, 100, 11):
            assert calc.multiply_numbers(a, b) == a * b
    rng = random.Random(2)
    for _ in range(30):
        a, b = rng.randrange(10**8), rng.randrange(10**8)
        assert calc.multiply_numbers(a, b) == a * b
    assert calc.multiply_numbers(48213, 9917) == 48213 * 9917
    assert calc.trace_tokens > 0
    assert set(calc.ram.state.unique().tolist()) <= {0.0, 1.0}


def test_calculator_rejects_operands_that_do_not_fit():
    with pytest.raises(ValueError):
        CompiledCalculator(max_digits=3).add_numbers(1000, 1)
