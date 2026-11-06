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


def gate(index: int, candidates: list[int], dtype: torch.dtype = torch.bfloat16) -> int:
    """Multiplicative selection (as in a gated MLP): one-hot(index) . candidates. Exact for integers
    below 256 in bfloat16, since only one product is nonzero."""
    onehot = torch.nn.functional.one_hot(torch.tensor(index), len(candidates)).to(dtype)
    return int((onehot @ torch.tensor(candidates, dtype=dtype)).item())


class CompiledMachine:
    """A stored-program machine whose whole step is compiled: the program, data, registers and program
    counter live in a 256-byte :class:`DeltaRAM`; fetch addresses, register addresses, indexed
    addresses, opcode decode, the ALU and the branch decision are :class:`LookupMLP` tables; the choice
    between datapaths is a multiplicative :func:`gate`. The host only repeats ``step()`` until the decoded
    instruction says HALT (as a decoder repeats forward passes until end-of-sequence); it contains no
    task logic. Values in registers must stay below 64.

    Memory map: program at [0, 128) (32 instructions of 4 bytes: op, a, b, c), operand A digits at
    [128, 144), B at [144, 160), result at [160, 192), registers r0..r15 at [192, 208), scratch 254,
    program counter 255. Bases for indexed access: 0 -> A, 1 -> B, 2 -> R.

    Instructions: 0 HALT; 1 LDI r,imm; 2 LDX r,base,ri (r = mem[base + ri]); 3 STX r,base,ri;
    4 ADDD r,x,y (r, r15 = digit sum of x, y and carry r15); 5 MULD r,x,y (r, r14 = units, tens of x*y);
    6 INC r; 7 DEC r; 8 JZ r,t; 9 JMP _,t; 10 MOV r,x; 11 JNZ r,t.
    """

    A, B, R, REGS, SCRATCH, PC = 128, 144, 160, 192, 254, 255
    CARRY, HI = 15, 14

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        self.dtype = dtype
        self.ram = DeltaRAM(heads=8, rows=32, dtype=dtype)
        reg = lambda x: self.REGS + x % 16  # noqa: E731
        bases = {0: self.A, 1: self.B, 2: self.R}
        self.fetch_addr = LookupMLP([32, 4], [256], lambda pc, k: 4 * pc + k, dtype)
        self.reg_addr = LookupMLP([64], [256], reg, dtype)
        self.index_addr = LookupMLP([64, 64], [256], lambda b, i: bases[b] + i if b in bases and i < 32 else self.SCRATCH, dtype)
        self.digit = LookupMLP([64], [10], lambda v: v if v < 10 else 0, dtype)
        self.acc = LookupMLP([10, 10, 10], [10, 10], lambda x, y, c: ((x + y + c) % 10, (x + y + c) // 10), dtype)
        self.mul = LookupMLP([10, 10], [10, 10], lambda x, y: ((x * y) % 10, (x * y) // 10), dtype)
        self.inc = LookupMLP([64], [64], lambda v: (v + 1) % 64, dtype)
        self.dec = LookupMLP([64], [64], lambda v: (v - 1) % 64, dtype)
        self.is_zero = LookupMLP([64], [2], lambda v: int(v == 0), dtype)
        # opcode -> (first-write address source, first-write value source, second write, pc rule)
        decode = {0: (2, 8, 0, 4), 1: (0, 0, 0, 0), 2: (0, 1, 0, 0), 3: (1, 2, 0, 0), 4: (0, 3, 1, 0),
                  5: (0, 4, 2, 0), 6: (0, 5, 0, 0), 7: (0, 6, 0, 0), 8: (2, 8, 0, 2), 9: (2, 8, 0, 1),
                  10: (0, 7, 0, 0), 11: (2, 8, 0, 3)}
        self.decode = LookupMLP([16], [3, 9, 3, 5], lambda op: decode.get(op, decode[0]), dtype)
        # (pc rule, register is zero) -> 0 sequential, 1 jump target, 2 stay (halt)
        branch = lambda rule, z: [0, 1, 1 if z else 0, 0 if z else 1, 2][rule]  # noqa: E731
        self.branch = LookupMLP([5, 2], [3], branch, dtype)
        self.trace: list[tuple] = []
        self.steps = 0

    def _read(self, address: int) -> int:
        value = self.ram.read(address)
        self.trace += [("ADDR", address), ("VAL", value)]
        return value

    def _write(self, address: int, value: int) -> None:
        self.ram.write(address, value)
        self.trace += [("ADDR", address), ("VAL", value)]

    def boot(self, program: list[tuple[int, int, int, int]], a: list[int] = (), b: list[int] = (),
             registers: dict[int, int] | None = None) -> None:
        """Load the program and the inputs (digits least significant first) into a zeroed memory."""
        if len(program) > 32 or len(a) > 16 or len(b) > 16:
            raise ValueError("program or operands too large for the memory map")
        self.ram = DeltaRAM(heads=8, rows=32, dtype=self.dtype)
        for i, instruction in enumerate(program):
            for k, byte in enumerate(instruction):
                self.ram.write(4 * i + k, byte)
        for base, digits in ((self.A, a), (self.B, b)):
            for i, digit in enumerate(digits):
                self.ram.write(base + i, digit)
        for r, value in (registers or {}).items():
            self.ram.write(self.REGS + r, value)
        self.trace, self.steps = [], 0

    def step(self) -> bool:
        """Execute one instruction; returns False once the machine has halted."""
        pc = self._read(self.PC)
        op, a, b, c = (self._read(self.fetch_addr(pc, k)[0]) for k in range(4))
        w1_addr_sel, w1_val_sel, w2_sel, pc_rule = self.decode(op % 16)
        ra = self._read(self.reg_addr(a % 64)[0])
        rb = self._read(self.reg_addr(b % 64)[0])
        rc = self._read(self.reg_addr(c % 64)[0])
        carry = self._read(self.REGS + self.CARRY)
        x_addr = self.index_addr(b % 64, rc % 64)[0]
        x_val = self._read(x_addr)
        db, dc, dcarry = (self.digit(v % 64)[0] for v in (rb, rc, carry))
        add_d, add_c = self.acc(db, dc, dcarry)
        mul_lo, mul_hi = self.mul(db, dc)
        values = [b, x_val, ra, add_d, mul_lo, self.inc(ra % 64)[0], self.dec(ra % 64)[0], rb, 0]
        self._write(gate(w1_addr_sel, [self.reg_addr(a % 64)[0], x_addr, self.SCRATCH], self.dtype),
                    gate(w1_val_sel, values, self.dtype))
        self._write(gate(w2_sel, [self.SCRATCH, self.REGS + self.CARRY, self.REGS + self.HI], self.dtype),
                    gate(w2_sel, [0, add_c, mul_hi], self.dtype))
        choice = self.branch(pc_rule, self.is_zero(ra % 64)[0])[0]
        self._write(self.PC, gate(choice, [self.inc(pc % 64)[0], b % 32, pc], self.dtype))
        self.trace.append(("STEP",))
        self.steps += 1
        return choice != 2

    def run(self, max_steps: int = 100_000) -> int:
        while self.step():
            if self.steps >= max_steps:
                raise RuntimeError(f"no HALT within {max_steps} steps")
        return self.steps

    def result(self, digits: int = 32) -> int:
        return int("".join(str(self.ram.read(self.R + i)) for i in reversed(range(digits))))


# Programs for CompiledMachine (inputs: digits of A and B, r2 = len(B) or len, r13 = len(A)).
ADD_PROGRAM = [  # r2 = max(len A, len B); r1 = index; r15 = carry
    (8, 2, 8, 0),    # 0: JZ r2 -> 8
    (2, 3, 0, 1),    # 1: r3 = A[r1]
    (2, 4, 1, 1),    # 2: r4 = B[r1]
    (4, 5, 3, 4),    # 3: r5, carry = r3 + r4 + carry
    (3, 5, 2, 1),    # 4: R[r1] = r5
    (6, 1, 0, 0),    # 5: r1 += 1
    (7, 2, 0, 0),    # 6: r2 -= 1
    (9, 0, 0, 0),    # 7: JMP 0
    (3, 15, 2, 1),   # 8: R[r1] = carry
    (0, 0, 0, 0),    # 9: HALT
]
MUL_PROGRAM = [  # r1 = j, r2 = len B left, r3 = i, r4 = len A left, r5 = k = i + j, r13 = len A
    (8, 2, 29, 0),   # 0: outer: JZ r2 -> END
    (2, 7, 1, 1),    # 1: r7 = B[j]
    (1, 3, 0, 0),    # 2: i = 0
    (10, 4, 13, 0),  # 3: r4 = len A
    (10, 5, 1, 0),   # 4: k = j
    (8, 4, 26, 0),   # 5: inner: JZ r4 -> NEXTJ
    (2, 6, 0, 3),    # 6: r6 = A[i]
    (5, 8, 6, 7),    # 7: r8, r14 = units, tens of A[i] * B[j]
    (2, 9, 2, 5),    # 8: r9 = R[k]
    (4, 9, 9, 8),    # 9: r9, carry = R[k] + units (carry is 0 here)
    (3, 9, 2, 5),    # 10: R[k] = r9
    (10, 12, 5, 0),  # 11: r12 = k
    (6, 12, 0, 0),   # 12: r12 = k + 1
    (2, 9, 2, 12),   # 13: r9 = R[k + 1]
    (4, 9, 9, 14),   # 14: r9, carry = R[k + 1] + tens + carry
    (3, 9, 2, 12),   # 15: R[k + 1] = r9
    (8, 15, 22, 0),  # 16: ripple: JZ carry -> CONT
    (6, 12, 0, 0),   # 17: r12 += 1
    (2, 9, 2, 12),   # 18: r9 = R[r12]
    (4, 9, 9, 10),   # 19: r9, carry = R[r12] + 0 + carry (r10 is always 0)
    (3, 9, 2, 12),   # 20: R[r12] = r9
    (9, 0, 16, 0),   # 21: JMP ripple
    (6, 3, 0, 0),    # 22: CONT: i += 1
    (6, 5, 0, 0),    # 23: k += 1
    (7, 4, 0, 0),    # 24: r4 -= 1
    (9, 0, 5, 0),    # 25: JMP inner
    (6, 1, 0, 0),    # 26: NEXTJ: j += 1
    (7, 2, 0, 0),    # 27: r2 -= 1
    (9, 0, 0, 0),    # 28: JMP outer
    (0, 0, 0, 0),    # 29: END: HALT
]


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
