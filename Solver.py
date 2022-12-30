import numpy as np

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
    return int(''.join(map(str, evaluate(_hyp, _ans))), 3)


def entropies():
    entropies_calc = np.zeros(len(all_words), dtype=np.float64)
    for i, _hyp in enumerate(all_words):
        combs = [comb_index(_hyp, _ans) for _ans in hidden_words]
        proba = np.bincount(combs, minlength=3 ** 5) / len(hidden_words)
        entropies_calc[i] = -np.sum(proba * np.log2(proba,out=np.zeros_like(proba), where=(proba != 0)))
    return entropies_calc


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
    # execution time of 200s
    init()
    ans = np.random.choice(hidden_words)

    for cnt in range(6):
        print(f'Round {cnt + 1}:')
        hyp = guess() if cnt else 'soare'
        print(f'Guess: {hyp} evaluated as {evaluate(hyp, ans)}')

        if all(evaluate(hyp, ans) == 2):
            print('Correct!')
            break

        if cnt == 5:
            print(f'The answer was {ans}.', "Failure", sep='\n')
            break

        update(hyp, evaluate(hyp, ans))