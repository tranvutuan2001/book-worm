"""
xgrammar-based constrained-decoding logits processor for mlx_lm.

The :class:`XGrammarMLXLogitsProcessor` wraps xgrammar's ``GrammarMatcher`` so
it can be passed directly to ``mlx_lm.generate`` (or ``stream_generate``) via
the ``logits_processors`` keyword argument.

Usage::

    processor = make_json_schema_logits_processor(tokenizer, json_schema_str)
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        logits_processors=[processor] if processor else None,
        ...
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import xgrammar as xgr


import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx


class XGrammarMLXLogitsProcessor:
    """
    Stateful logits processor for mlx_lm that enforces an xgrammar grammar.

    The processor conforms to the ``logits_processors`` protocol expected by
    ``mlx_lm.generate_step``::

        Callable[[mx.array, mx.array], mx.array]
                  ^^^^^^^^  ^^^^^^^^  ^^^^^^^^^
                  tokens    logits    masked logits

    The ``tokens`` argument is the *full* token sequence generated so far
    (prompt + new tokens).  On every call we advance the :class:`GrammarMatcher`
    by the tokens added since the previous call, fill the next-token bitmask,
    then zero-out (−∞) all disallowed positions in *logits*.

    Notes
    -----
    - Create a **new instance** for each ``generate()`` call; the internal
      matcher state is not reusable.
    - Only the vocabulary positions ``[0, vocab_size)`` are masked; any extra
      positions (padding) are left untouched.
    """

    def __init__(self, compiled_grammar: Any) -> None:
        import xgrammar as xgr

        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size: int = compiled_grammar.tokenizer_info.vocab_size

        # Pre-allocate the int32 bitmask buffer (shape: [1, ceil(vocab/32)])
        bitmask_shape = xgr.get_bitmask_shape(1, self._vocab_size)
        self._bitmask = np.zeros(bitmask_shape, dtype=np.int32)

        # Number of tokens already accepted by the matcher from previous calls.
        self._prev_len: int = 0

    def __call__(self, tokens: "mx.array", logits: "mx.array") -> "mx.array":
        import mlx.core as mx

        # ---- 1. Advance the matcher with any new tokens ----
        token_ids: list[int] = tokens.tolist()
        for tok in token_ids[self._prev_len :]:
            if self._matcher.is_terminated():
                break
            self._matcher.accept_token(int(tok))
        self._prev_len = len(token_ids)

        # If grammar has been fully matched, do not restrict further.
        if self._matcher.is_terminated():
            return logits

        # ---- 2. Fill the bitmask for the next token ----
        self._matcher.fill_next_token_bitmask(self._bitmask, 0)

        # ---- 3. Unpack packed int32 bits → bool mask (shape: [vocab_size]) ----
        # view as uint8 then unpackbits with little-endian bit order so that
        # bit index i corresponds to token id i.
        unpacked: np.ndarray = np.unpackbits(
            self._bitmask.view(np.uint8), axis=-1, bitorder="little"
        )  # shape: (1, 4 * cols)
        allowed: np.ndarray = unpacked[0, : self._vocab_size].astype(bool)

        # ---- 4. Apply mask: disallowed positions → −∞ ----
        mask_mx = mx.array(allowed)  # bool, shape (vocab_size,)

        # logits may have shape (vocab_size,) or (..., vocab_size)
        vocab_slice = logits[..., : self._vocab_size]
        masked = mx.where(mask_mx, vocab_slice, mx.array(float("-inf"), dtype=logits.dtype))

        if logits.shape[-1] > self._vocab_size:
            # Preserve any extra padding positions untouched.
            tail = logits[..., self._vocab_size :]
            return mx.concatenate([masked, tail], axis=-1)

        return masked


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_json_schema_logits_processor(
    tokenizer: Any,
    json_schema: str,
    vocab_size: int | None = None,
) -> XGrammarMLXLogitsProcessor | None:
    """
    Build an :class:`XGrammarMLXLogitsProcessor` that constrains generation to
    the given *json_schema*.

    Parameters
    ----------
    tokenizer:
        The HuggingFace (or mlx_lm ``TokenizerWrapper``) tokenizer for the
        model that will be used for generation.
    json_schema:
        A JSON Schema string (or dict serialised to string) that every
        generated token sequence must conform to.
    vocab_size:
        Override for the model's vocabulary size.  When ``None`` the value is
        derived from ``len(tokenizer)``, which includes special/added tokens
        and matches the model's actual embedding matrix size.

    Returns
    -------
    XGrammarMLXLogitsProcessor | None
        ``None`` when xgrammar is not installed or the processor cannot be
        constructed (a warning is logged in both cases).
    """
    # Unwrap mlx_lm TokenizerWrapper if necessary so xgrammar gets a real
    # HuggingFace PreTrainedTokenizer.
    raw_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)

    # Resolve vocabulary size.
    # IMPORTANT: use len(raw_tokenizer) — not tokenizer.vocab_size — because
    # vocab_size is the *base* size and excludes added/special tokens.
    # len(tokenizer) reflects the full vocabulary (base + added tokens),
    # which matches the model's embedding matrix and avoids xgrammar's
    # "token id X is out of range" warning.
    if vocab_size is None:
        try:
            vocab_size = len(raw_tokenizer)
        except TypeError:
            # Fallback: some tokenizer wrappers don't support len()
            if hasattr(raw_tokenizer, "vocab_size"):
                vocab_size = raw_tokenizer.vocab_size
            else:
                vocab_size = len(raw_tokenizer.get_vocab())

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        raw_tokenizer, vocab_size=vocab_size
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(json_schema)
    return XGrammarMLXLogitsProcessor(compiled_grammar)