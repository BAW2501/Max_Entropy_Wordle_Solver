import numpy as np
from tqdm import tqdm
import ctypes


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
    if _hyp[0] == _ans[0]:
        index += 2 * (3 ** 0)
        _ans[0] = '0'
    elif _hyp[0] in _ans:
        index += 1 * (3 ** 0)
    if _hyp[1] == _ans[1]:
        index += 2 * (3 ** 1)
        _ans[1] = '0'
    elif _hyp[1] in _ans:
        index += 1 * (3 ** 1)
    if _hyp[2] == _ans[2]:
        index += 2 * (3 ** 2)
        _ans[2] = '0'
    elif _hyp[2] in _ans:
        index += 1 * (3 ** 2)
    if _hyp[3] == _ans[3]:
        index += 2 * (3 ** 3)
        _ans[3] = '0'
    elif _hyp[3] in _ans:
        index += 1 * (3 ** 3)
    if _hyp[4] == _ans[4]:
        index += 2 * (3 ** 4)
    elif _hyp[4] in _ans:
        index += 1 * (3 ** 4)
    return index


def calc_proba(words, _hyp):
    probas = np.zeros(3 ** 5, dtype=np.float64)
    for _ans in words:
        comb = comb_index(_hyp, _ans)
        probas[comb] += 1
    return probas / len(words)


def calc_proba_python(words, _hyp):
    words_count = len(words)
    probas = np.zeros(3 ** 5, dtype=np.float64)
    words_ptr = (ctypes.c_char_p * words_count)()
    words_ptr[:] = words  # [word.encode() for word in words]
    probas_ptr = probas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib = ctypes.cdll.LoadLibrary('./fast_comb_proba.dll')
    lib.calc_proba(words_ptr, words_count, _hyp, probas_ptr)

    return probas


def entropies():
    entropies_calc = np.zeros(len(all_words), dtype=np.float64)
    for i, _hyp in tqdm(enumerate(all_words), total=len(all_words)):
        try:
            proba = calc_proba_python(hidden_words, _hyp)
        except Exception as e:
            proba = calc_proba(hidden_words, _hyp)
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
    all_words = [word.encode() for word in all_words]
    hidden_words = np.loadtxt("data/english-hidden.txt", dtype=str)
    hidden_words = [word.encode() for word in hidden_words]


if __name__ == "__main__":
    # execution time of 31s with calc_proba
    # execution time of 8s with the calc_proba_python that calls compiled c code
    init()
    ans = np.random.choice(hidden_words)

    for cnt in range(6):
        print(f'Round {cnt + 1}:')
        hyp = guess()  # if cnt != 0 else 'soare'
        print(f'Guess: {hyp.decode()} evaluated as {evaluate(hyp, ans)}')

        if all(evaluate(hyp, ans) == 2):
            print('Correct!')
            break

        if cnt == 5:
            print(f'The answer was {ans.decode()}.', "Failure", sep='\n')
            break

        update(hyp, evaluate(hyp, ans))
