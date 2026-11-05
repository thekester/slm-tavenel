"""Compiled-computer primitives (Qwenputer-style, see experiments/AXIS-compiled-computer-qwenputer-2026-11-05.md).

Memory is the state of a gated delta-rule layer (Yang et al., 2025, "Gated Delta Networks"):

    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,    o_t = S_t q_t

with S of shape (d_v, d_k) per head. With alpha = beta = 1 and a one-hot key e_r, the update replaces
column r of S by v_t and leaves every other column untouched, and a one-hot query e_r reads column r back:
an exact addressable RAM. alpha = 0 for one step clears a head; alpha = 1/2 halves it. Nothing here is
learned; weights and gates are set by construction, and the arithmetic stays exact in bfloat16 because
every product involves a 0/1 factor and every sum has at most two terms.
"""
from __future__ import annotations

import torch


def gated_delta_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                     beta: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential gated delta rule.

    q, k: (B, H, T, d_k); v: (B, H, T, d_v); alpha, beta: (B, H, T); state: (B, H, d_v, d_k) or None.
    Returns outputs (B, H, T, d_v) read after each update, and the final state.
    """
    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    if state is None:
        state = torch.zeros(b, h, d_v, d_k, dtype=v.dtype, device=v.device)
    outputs = []
    for i in range(t):
        k_i, v_i = k[:, :, i], v[:, :, i]
        a_i, b_i = alpha[:, :, i, None, None], beta[:, :, i, None, None]
        recalled = torch.einsum("bhvk,bhk->bhv", state, k_i)  # S k: what the key currently addresses
        state = a_i * (state - b_i * torch.einsum("bhv,bhk->bhvk", recalled, k_i)) \
            + b_i * torch.einsum("bhv,bhk->bhvk", v_i, k_i)
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, :, i]))
    return torch.stack(outputs, dim=2), state


def byte_to_bits(value: int) -> list[int]:
    if not 0 <= value < 256:
        raise ValueError(f"not a byte: {value}")
    return [(value >> i) & 1 for i in range(8)]


def bits_to_byte(bits: torch.Tensor) -> int:
    rounded = bits.float().round().long().tolist()
    if any(b not in (0, 1) for b in rounded):
        raise ValueError(f"not a bit vector: {bits.tolist()}")
    return sum(b << i for i, b in enumerate(rounded))


class DeltaRAM:
    """Byte-addressable memory held in gated delta-rule state: ``heads`` x ``rows`` bytes, one byte per
    column (8 bit lanes). Address a maps to head a % heads and row a // heads, as in Qwenputer (low bits
    pick the head). Every operation is one token step of :func:`gated_delta_scan`."""

    def __init__(self, heads: int, rows: int, dtype: torch.dtype = torch.bfloat16):
        self.heads, self.rows, self.dtype = heads, rows, dtype
        self.state = torch.zeros(1, heads, 8, rows, dtype=dtype)

    @property
    def size(self) -> int:
        return self.heads * self.rows

    def _locate(self, address: int) -> tuple[int, int]:
        if not 0 <= address < self.size:
            raise IndexError(f"address {address} outside 0..{self.size - 1}")
        return address % self.heads, address // self.heads

    def _step(self, head: int, row: int | None, value: list[int] | None, alpha: float, beta: float,
              read_row: int | None = None) -> torch.Tensor:
        """One token step: only ``head`` is active; the other heads use alpha = 1, beta = 0 (unchanged)."""
        k = torch.zeros(1, self.heads, 1, self.rows, dtype=self.dtype)
        q = torch.zeros_like(k)
        v = torch.zeros(1, self.heads, 1, 8, dtype=self.dtype)
        alphas = torch.ones(1, self.heads, 1, dtype=self.dtype)
        betas = torch.zeros(1, self.heads, 1, dtype=self.dtype)
        if row is not None:
            k[0, head, 0, row] = 1
        if value is not None:
            v[0, head, 0] = torch.tensor(value, dtype=self.dtype)
        if read_row is not None:
            q[0, head, 0, read_row] = 1
        alphas[0, head, 0], betas[0, head, 0] = alpha, beta
        out, self.state = gated_delta_scan(q, k, v, alphas, betas, self.state)
        return out[0, head, 0]

    def write(self, address: int, value: int) -> None:
        head, row = self._locate(address)
        self._step(head, row, byte_to_bits(value), alpha=1.0, beta=1.0)

    def read(self, address: int) -> int:
        head, row = self._locate(address)
        return bits_to_byte(self._step(head, None, None, alpha=1.0, beta=0.0, read_row=row))

    def clear_head(self, head: int) -> None:
        self._step(head, None, None, alpha=0.0, beta=0.0)

    def halve_head(self, head: int) -> None:
        self._step(head, None, None, alpha=0.5, beta=0.0)


class LookupMLP(torch.nn.Module):
    """A compiled two-layer MLP that evaluates a finite function exactly.

    Inputs are one-hot vectors, one per argument (sizes ``arities``). The hidden layer has one ReLU unit
    per input combination, computing AND of its one-hot bits as relu(sum of the selected bits - (n - 1));
    the output layer holds the table, one column per combination, with one one-hot block per output
    (sizes ``outputs``). Hidden values are 0 or 1 and each output sums exactly one nonzero term, so the
    result is exact in bfloat16. This is the "MLP as lookup table" construction used for ALUs in
    compiled transformers (Tracr, Lindner et al., 2023; Qwenputer).
    """

    def __init__(self, arities: list[int], outputs: list[int], table, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        import itertools

        self.arities, self.outputs = list(arities), list(outputs)
        combos = list(itertools.product(*(range(a) for a in arities)))
        offsets = [sum(arities[:i]) for i in range(len(arities))]
        out_offsets = [sum(outputs[:i]) for i in range(len(outputs))]
        self.up = torch.nn.Linear(sum(arities), len(combos), dtype=dtype)
        self.down = torch.nn.Linear(len(combos), sum(outputs), bias=False, dtype=dtype)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.fill_(-(len(arities) - 1))
            self.down.weight.zero_()
            for j, combo in enumerate(combos):
                for value, offset in zip(combo, offsets):
                    self.up.weight[j, offset + value] = 1
                result = table(*combo)
                result = result if isinstance(result, tuple) else (result,)
                for value, offset, size in zip(result, out_offsets, outputs):
                    if not 0 <= value < size:
                        raise ValueError(f"table value {value} outside 0..{size - 1}")
                    self.down.weight[offset + value, j] = 1
        self.requires_grad_(False)

    def forward(self, onehots: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(onehots)))

    def __call__(self, *values: int) -> tuple[int, ...]:
        dtype = self.up.weight.dtype
        x = torch.cat([torch.nn.functional.one_hot(torch.tensor(v), a).to(dtype) for v, a in zip(values, self.arities)])
        y = self.forward(x)
        result, start = [], 0
        for size in self.outputs:
            block = y[start:start + size]
            if sorted(block.tolist()) != [0.0] * (size - 1) + [1.0]:
                raise RuntimeError(f"non one-hot output {block.tolist()}")
            result.append(int(block.argmax()))
            start += size
        return tuple(result)


class CompiledCalculator:
    """Exact decimal addition and multiplication by a compiled machine: operands and result are digit
    bytes in a :class:`DeltaRAM`, every digit operation goes through a :class:`LookupMLP`, and each
    machine step is logged as a trace (``("ADDR", a)``, ``("VAL", v)``, ``("STEP",)``), like Qwenputer's
    trace tokens. Nothing is learned. ``trace_tokens`` counts the cost of the last operation.

    Memory layout (digits least significant first): operand a at [0, max_digits), operand b at
    [max_digits, 2 max_digits), result at [2 max_digits, 4 max_digits)."""

    def __init__(self, max_digits: int = 16, dtype: torch.dtype = torch.bfloat16):
        self.max_digits = max_digits
        self.ram = DeltaRAM(heads=8, rows=(4 * max_digits + 7) // 8, dtype=dtype)
        # (a, b, carry) -> (sum digit, carry) for addition; (a, b) -> (units, tens) for digit products;
        # (x, y, carry) with x, y digits and carry 0..9 -> (digit, carry) for multiply-accumulate.
        self.add = LookupMLP([10, 10, 2], [10, 2], lambda a, b, c: ((a + b + c) % 10, (a + b + c) // 10), dtype)
        self.mul = LookupMLP([10, 10], [10, 10], lambda a, b: ((a * b) % 10, (a * b) // 10), dtype)
        self.acc = LookupMLP([10, 10, 10], [10, 10], lambda x, y, c: ((x + y + c) % 10, (x + y + c) // 10), dtype)
        self.trace: list[tuple] = []

    @property
    def trace_tokens(self) -> int:
        return len(self.trace)

    def _write(self, address: int, value: int) -> None:
        self.ram.write(address, value)
        self.trace += [("ADDR", address), ("VAL", value)]

    def _read(self, address: int) -> int:
        value = self.ram.read(address)
        self.trace += [("ADDR", address), ("VAL", value)]
        return value

    def _load(self, a: int, b: int) -> None:
        for name, value in (("a", a), ("b", b)):
            if value < 0 or len(str(value)) > self.max_digits:
                raise ValueError(f"operand {name}={value} outside 0..10^{self.max_digits} - 1")
        self.trace = []
        for i in range(4 * self.max_digits):
            self.ram.write(i, 0)  # reset: one overwrite per cell (not traced, part of the boot)
        for base, value in ((0, a), (self.max_digits, b)):
            for i, digit in enumerate(reversed(str(value))):
                self._write(base + i, int(digit))

    def _result(self) -> int:
        digits = [self.ram.read(2 * self.max_digits + i) for i in range(2 * self.max_digits)]
        return int("".join(str(d) for d in reversed(digits)))

    def _length(self, value: int) -> int:
        return len(str(value))

    def add_numbers(self, a: int, b: int) -> int:
        self._load(a, b)
        n, carry, out = self.max_digits, 0, 2 * self.max_digits
        for i in range(max(self._length(a), self._length(b))):
            digit, carry = self.add(self._read(i), self._read(n + i), carry)
            self._write(out + i, digit)
            self.trace.append(("STEP",))
        if carry:
            self._write(out + max(self._length(a), self._length(b)), carry)
        return self._result()

    def multiply_numbers(self, a: int, b: int) -> int:
        """Schoolbook multiplication: for every digit pair, add the product's units and tens into the
        result in place, then propagate the carry with the accumulate table."""
        self._load(a, b)
        n, out = self.max_digits, 2 * self.max_digits
        for j in range(self._length(b)):
            b_j = self._read(n + j)
            for i in range(self._length(a)):
                units, tens = self.mul(self._read(i), b_j)
                position, carry = i + j, 0
                for addend in (units, tens):
                    digit, carry = self.acc(self._read(out + position), addend, carry)
                    self._write(out + position, digit)
                    position += 1
                while carry:  # ripple the remaining carry
                    digit, carry = self.acc(self._read(out + position), 0, carry)
                    self._write(out + position, digit)
                    position += 1
                self.trace.append(("STEP",))
        return self._result()
