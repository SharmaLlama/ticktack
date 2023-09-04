#from __future__ import annotations
import __future__

print(__future__.annotations)
#print(annotations)

class A(object):
    def __init__(self, a) -> A:
        self.a = a
