import time
from typing import Optional
import numpy as np
from joblib import Parallel, delayed

WORD_LEN = 5
GREEN, YELLOW = 2, 1


class WordleSolver:
    def __init__(self):
        self.all_words = np.loadtxt("data/english-all.txt", dtype=str)
        self.hidden_words = np.loadtxt("data/english-hidden.txt", dtype=str)

    @staticmethod
    def evaluate(hyp: str, ans: str) -> str:
        return np.base_repr(WordleSolver.comb_index(hyp, ans), 3).rjust(5, "0")

    @staticmethod
    def comb_index(hyp: str, ans: str) -> int:
        remaining: list[Optional[str]] = list(ans)
        index = 0
        for i in range(WORD_LEN):  # greens first
            if hyp[i] == ans[i]:
                index += GREEN * 3 ** (WORD_LEN - 1 - i)
                remaining[i] = None
        for i in range(WORD_LEN):  # then yellows, on what's left
            if hyp[i] != ans[i] and hyp[i] in remaining:
                index += YELLOW * 3 ** (WORD_LEN - 1 - i)
                remaining[remaining.index(hyp[i])] = None
        return index

    def calc_entropy(self, hyp: str) -> np.float32:
        combs = np.fromiter(
            (self.comb_index(hyp, ans) for ans in self.hidden_words),
            np.uint8,
            len(self.hidden_words),
        )
        probas = (np.bincount(combs) / len(self.hidden_words)).astype(np.float32)
        log_probas = np.log2(probas, where=0 < probas, out=0 * probas)
        return -np.sum(probas * log_probas)

    def entropies(self) -> np.ndarray:
        return np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(self.calc_entropy)(_hyp) for _hyp in self.all_words
            )
        )

    def guess(self) -> str:
        return self.all_words[np.argmax(self.entropies())]

    def update(self, hyp: str, tiles: str) -> None:
        candidates = [
            cand for cand in self.all_words if tiles == self.evaluate(hyp, cand)
        ]
        self.all_words = np.intersect1d(self.all_words, candidates)
        self.hidden_words = np.intersect1d(self.hidden_words, candidates)


if __name__ == "__main__":
    solver = WordleSolver()
    answer = np.random.choice(solver.hidden_words)  #'vomit' #

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
