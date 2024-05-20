import re
from sys import stdin

for line in stdin:
    for word in re.findall(r'\w+', line):
        print(f"{word}\t1")
