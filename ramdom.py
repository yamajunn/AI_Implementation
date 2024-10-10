import time

a = 1587123
c = 2390751
m = 2**32

print((a * int(time.time() * 1000) + c) % m / m)