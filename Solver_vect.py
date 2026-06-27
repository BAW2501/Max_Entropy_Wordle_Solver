import time
from typing import Optional
import numpy as np

WORD_LEN = 5
GREEN, YELLOW = 2, 1
N_PATTERNS = 3 ** WORD_LEN  # 243


class WordleSolver:
    """
    Entropy-maximizing Wordle solver.

    Vectorization idea: the pattern between a fixed guess and a fixed answer
    never changes between rounds, so it's wasteful to recompute it every
    round. Instead we build the full (guesses x candidate-answers) pattern
    matrix once -- lazily, on first use, not in __init__ -- and every round
    after that is just slicing + counting that cached matrix.
    """

    __slots__ = ("all_words", "hidden_words", "_word_to_idx", "_pattern", "_alive")

    def __init__(self):
        self.all_words = np.loadtxt("data/english-all.txt", dtype=str)
        self.hidden_words = np.loadtxt("data/english-hidden.txt", dtype=str)
        self._word_to_idx = {w: i for i, w in enumerate(self.all_words)}
        self._pattern = None  # built lazily -- see `pattern_matrix` below
        self._alive = np.ones(len(self.hidden_words), dtype=bool)

    # ---- scalar reference implementation (used once per round, to score the real answer) ----

    @staticmethod
    def comb_index(hyp: str, ans: str) -> int:
        remaining :list[Optional[str]] = list(ans)
        for i in range(WORD_LEN):  # lock in greens first
            if hyp[i] == ans[i]:
                remaining[i] = None
        index = 0
        for i in range(WORD_LEN):
            if hyp[i] == ans[i]:
                index += GREEN * 3 ** (WORD_LEN - 1 - i)
            elif hyp[i] in remaining:  # yellow, consuming what's left
                index += YELLOW * 3 ** (WORD_LEN - 1 - i)
                remaining[remaining.index(hyp[i])] = None
        return index

    @staticmethod
    def evaluate(hyp: str, ans: str) -> str:
        return np.base_repr(WordleSolver.comb_index(hyp, ans), 3).rjust(WORD_LEN, "0")

    # ---- vectorized pattern matrix: the part worth studying ----

    @property
    def pattern_matrix(self) -> np.ndarray:
        """(len(all_words), len(hidden_words)) uint8 matrix, built once, on first access."""
        if self._pattern is None:
            self._pattern = self._build_pattern_matrix()
        return self._pattern

    def _build_pattern_matrix(self) -> np.ndarray:
        guesses = self._to_codes(self.all_words)      # (G, 5) uint8 letter codes
        answers = self._to_codes(self.hidden_words)   # (H, 5) uint8 letter codes
        weights = (3 ** np.arange(WORD_LEN - 1, -1, -1)).astype(np.uint8)

        # eq[g, h, i, j] = guess g's letter i == answer h's letter j
        eq = guesses[:, None, :, None] == answers[None, :, None, :]
        out = np.zeros((len(guesses), len(answers)), dtype=np.uint8)

        for i in range(WORD_LEN):  # greens: same-position matches
            m = eq[:, :, i, i]
            out += m.astype(np.uint8) * (GREEN * weights[i])
            keep = ~m[:, :, None]
            eq[:, :, :, i] &= keep  # this answer letter is now used up
            eq[:, :, i, :] &= keep  # this guess letter is now used up

        for i in range(WORD_LEN):  # yellows: cross-position matches on what's left
            for j in range(WORD_LEN):
                if i == j:
                    continue
                m = eq[:, :, i, j]
                out += m.astype(np.uint8) * (YELLOW * weights[i])
                keep = ~m[:, :, None]
                eq[:, :, :, j] &= keep
                eq[:, :, i, :] &= keep

        return out.astype(np.uint8)

    @staticmethod
    def _to_codes(words: np.ndarray) -> np.ndarray:
        return np.array([[ord(c) for c in w] for w in words], dtype=np.uint8)

    # ---- entropy / guessing ----

    def entropies(self) -> np.ndarray:
        sub = self.pattern_matrix[:, self._alive]          # triggers the lazy build on first call
        n_alive = sub.shape[1]
        counts = np.zeros((len(self.all_words), N_PATTERNS), dtype=np.int32)
        rows = np.repeat(np.arange(len(self.all_words)), n_alive)
        np.add.at(counts, (rows, sub.ravel()), 1)
        probas = (counts / n_alive).astype(np.float32)     # float32 from here on, no upcast warning
        log_probas = np.log2(probas, where=probas > 0, out=np.zeros_like(probas))
        return -np.sum(probas * log_probas, axis=1)

    def guess(self) -> str:
        candidates = self.hidden_words[self._alive]
        if len(candidates) == 1:
            return candidates[0]  # nothing left to learn -- just guess it
        return self.all_words[np.argmax(self.entropies())]

    def update(self, hyp: str, tiles: str) -> None:
        p = int(tiles, 3)
        row = self.pattern_matrix[self._word_to_idx[hyp]]
        self._alive &= row == p


if __name__ == "__main__":
    solver = WordleSolver()
    answer = np.random.choice(solver.hidden_words)

    start = time.time()
    for cnt in range(6):
        print(f"Round {cnt + 1}:")
        guess = solver.guess()
        tiles = solver.evaluate(guess, answer)
        print(f"Guess: {guess} evaluated as {tiles}")
        if str(answer) == str(guess):
            print("Correct!")
            break
        if cnt == 5:
            print(f"The answer was {answer}.", "Failure", sep="\n")
            break
        solver.update(guess, tiles)
    print(f"Execution time: {time.time() - start:.2f}s")