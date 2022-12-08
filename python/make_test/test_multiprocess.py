import multiprocessing
import psutil
from multiprocessing import Process, Process, Queue
import os
import time
num_processes = psutil.cpu_count(logical=False)
x = 19
def square(n, queue):
    a = {}

    a["x"] = x**n
    time.sleep(10)
    # print(x)
    queue.put(a)
if __name__ == '__main__':
    res = []
    # print(num_processes)
    for _ in range(num_processes):
        q = Queue()
        p = Process(target=square, args=(_,q))
        res.append((p, q))
        p.start()
    for p, q in res:
        p.join()
        print(q.get())

    
