
import sys
import procgen
print(f"Procgen file: {procgen.__file__}")
print(f"Procgen dir: {dir(procgen)}")
try:
    from procgen import ProcgenGym3Env
    print("ProcgenGym3Env found")
except ImportError:
    print("ProcgenGym3Env NOT found")

