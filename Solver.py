import time
import numpy as np
from joblib import Parallel, delayed


def evaluate(_hyp, _ans):
    _tiles = np.zeros(5, dtype=int)
    _ans = list(_ans)
    for i in range(5):
        if _hyp[i] == _ans[i]:
            _tiles[i] = 2
            _ans[i] = '0'
        elif _hyp[i] in _ans:
            _tiles[i] = 1
    return _tiles


def comb_index(_hyp, _ans):
    _ans = list(_ans)
    index = 0
    for i in range(5):
        if _hyp[i] == _ans[i]:
            index += 2 * (3 ** i)
            _ans[i] = '0'
        elif _hyp[i] in _ans:
            index += 1 * (3 ** i)
    return index


def calc_entropy(words, _hyp):
    probas = np.zeros(3 ** 5, dtype=np.float64)
    for _ans in words:
        comb = comb_index(_hyp, _ans)
        probas[comb] += 1
    probas = probas / len(words)
    return -np.sum(probas* np.log2(probas, out=np.zeros_like(probas), where=(probas != 0)))


def entropies():
    return np.array(Parallel(n_jobs=-1,verbose=0)(delayed(calc_entropy)(hidden_words, _hyp) for _hyp in all_words
    ))


def guess():
    return all_words[np.argmax(entropies())]


def update(_hyp, _tiles):
    global all_words, hidden_words
    candidates = [cand for cand in all_words if all(_tiles == evaluate(_hyp, cand))]
    all_words = np.intersect1d(all_words, candidates)
    hidden_words = np.intersect1d(hidden_words, candidates)


def init():
    global all_words, hidden_words
    all_words = np.loadtxt("data/english-all.txt", dtype=str)
    hidden_words = np.loadtxt("data/english-hidden.txt", dtype=str)


if __name__ == "__main__":
    # execution time of 11s with calc_proba
    init()
    ans = np.random.choice(hidden_words)
    start = time.time()
    for cnt in range(6):
        print(f'Round {cnt + 1}:')
        hyp = guess()  # if cnt != 0 else 'soare'
        print(f'Guess: {hyp} evaluated as {evaluate(hyp, ans)}')

        if all(evaluate(hyp, ans) == 2):
            print('Correct!')
            break

        if cnt == 5:
            print(f'The answer was {ans}.', "Failure", sep='\n')
            break

        update(hyp, evaluate(hyp, ans))

    print(f'Execution time: {time.time() - start:.2f}s')