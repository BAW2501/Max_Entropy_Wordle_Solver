int comb_index(const char *__restrict _hyp,char *__restrict _ans) { 
    // clang -shared -o fast_comb_proba.dll fast_comb_proba.c -Ofast
  int index = 0, power_of_3 = 1, i;
  for (i = 0; i < 5; i++, power_of_3 *= 3) {
    if (_hyp[i] == _ans[i]) {
      index += 2 * power_of_3;
      _ans[i] = '0';
    } else if (_ans[0] == _hyp[i] || _ans[1] == _hyp[i] || _ans[2] == _hyp[i] || _ans[3] == _hyp[i] || _ans[4] == _hyp[i])
      index += 1 * power_of_3;
  }
  return index;
}

void calc_proba(char **__restrict words, int words_count,const char*__restrict _hyp, double *__restrict probas)
{
    int i;
    for (i = 0; i < words_count; i++)
        probas[comb_index(_hyp, words[i])] += 1 / (double) words_count;
}
