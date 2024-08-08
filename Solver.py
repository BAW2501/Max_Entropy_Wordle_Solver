#import time
import numpy as np
from joblib import Parallel, delayed

class WordleSolver:
    def __init__(self):
        self.all_words = np.loadtxt("data/english-all.txt", dtype=str)
        self.hidden_words = np.loadtxt("data/english-hidden.txt", dtype=str)

    def evaluate(self, hyp: str, ans: str) -> str:
        return np.base_repr(self.comb_index(hyp, ans), 3).rjust(5, '0')

    def comb_index(self, hyp: str, ans: str) -> int: # inline for efficiency
        ans_list = list(ans)
        index = 0
        for i in range(5):
            if hyp[i] == ans_list[i]:
                index += 2 * (3 ** i)
                ans_list[i] = "0"
            elif hyp[i] in ans_list:
                index += 1 * (3 ** i)
        return index

    def calc_entropy(self, hyp: str) -> np.float32:
        combs = np.fromiter((self.comb_index(hyp, ans) for ans in self.hidden_words), np.uint8, len(self.hidden_words))
        probas = np.bincount(combs) / len(self.hidden_words)
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
        candidates = [cand for cand in self.all_words if tiles == self.evaluate(hyp, cand)]
        self.all_words = np.intersect1d(self.all_words, candidates)
        self.hidden_words = np.intersect1d(self.hidden_words, candidates)

if __name__ == "__main__":
    solver = WordleSolver()
    answer = np.random.choice(solver.hidden_words)
    
    #start = time.time()
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

    #print(f"Execution time: {time.time() - start:.2f}s")