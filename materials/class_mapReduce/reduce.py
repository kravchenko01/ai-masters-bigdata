from sys import stdin

words = dict()
for line in stdin:
    w, n = line.split()
    n = int(n)
    if w in words.keys():
        words[w] += n
    else:
        words[w] = n

for token, count in words.items():
    print(f'{token}\t{count}')

