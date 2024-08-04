# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
import time

np.import_array()

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t DTYPE_float_t

cdef vector[string] all_words
cdef vector[string] hidden_words

cdef np.ndarray[DTYPE_t, ndim=1] evaluate(string hyp, string ans):
    cdef np.ndarray[DTYPE_t, ndim=1] tiles = np.zeros(5, dtype=np.int_)
    cdef string temp_ans = ans
    cdef int i
    
    for i in range(5):
        if hyp[i] == temp_ans[i]:
            tiles[i] = 2
            temp_ans = temp_ans[:i] + b'0' + temp_ans[i+1:]
    
    for i in range(5):
        if tiles[i] == 0:
            pos = temp_ans.find(hyp[i])
            if pos != -1:
                tiles[i] = 1
                temp_ans = temp_ans[:pos] + b'0' + temp_ans[pos+1:]
    
    return tiles

cdef double calc_entropy(vector[string]& words, string hyp):
    cdef np.ndarray[DTYPE_float_t, ndim=1] counts = np.zeros(243, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] eval_result
    cdef double total = len(words)
    cdef double prob, entropy = 0
    cdef int i, index
    
    for ans in words:
        eval_result = evaluate(hyp, ans)
        index = eval_result[0] * 81 + eval_result[1] * 27 + eval_result[2] * 9 + eval_result[3] * 3 + eval_result[4]
        counts[index] += 1
    
    for i in range(243):
        if counts[i] > 0:
            prob = counts[i] / total
            entropy -= prob * np.log2(prob)
    
    return entropy

def guess():
    cdef string best_word
    cdef double max_entropy = -1
    cdef double entropy
    
    for word in all_words:
        entropy = calc_entropy(hidden_words, word)
        if entropy > max_entropy:
            max_entropy = entropy
            best_word = word
    
    return best_word

cdef bint is_word_in_vector(string word, vector[string]& vec):
    cdef size_t i
    for i in range(vec.size()):
        if vec[i] == word:
            return True
    return False

def update(string hyp, np.ndarray[DTYPE_t, ndim=1] tiles):
    global all_words, hidden_words
    cdef vector[string] new_all_words, new_hidden_words
    cdef np.ndarray[DTYPE_t, ndim=1] eval_result
    
    for word in all_words:
        eval_result = evaluate(hyp, word)
        if np.array_equal(tiles, eval_result):
            new_all_words.push_back(word)
    
    for word in hidden_words:
        if is_word_in_vector(word, new_all_words):
            new_hidden_words.push_back(word)
    
    all_words = new_all_words
    hidden_words = new_hidden_words

def init():
    global all_words, hidden_words
    all_words = np.loadtxt("data/english-all.txt", dtype='S5').tolist()
    hidden_words = np.loadtxt("data/english-hidden.txt", dtype='S5').tolist()

def main():
    init()
    cdef string ans = np.random.choice(hidden_words)
    cdef double start = time.time()
    cdef int cnt
    cdef string hyp
    cdef np.ndarray[DTYPE_t, ndim=1] eval_result
    
    for cnt in range(6):
        print(f'Round {cnt + 1}:')
        hyp = guess()
        eval_result = evaluate(hyp, ans)
        print(f'Guess: {hyp.decode("utf-8")} evaluated as {eval_result}')

        if np.all(eval_result == 2):
            print('Correct!')
            break

        if cnt == 5:
            print(f'The answer was {ans.decode("utf-8")}.', "Failure", sep='\n')
            break

        update(hyp, eval_result)

    print(f'Execution time: {time.time() - start:.2f}s')

if __name__ == "__main__":
    main()