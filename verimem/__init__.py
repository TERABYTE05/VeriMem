"""VeriMem as one command.

The package exists so that `python -m verimem "<claim>"` works. Everything it calls lives
in the topic packages (`agent`, `retrieval`, `experience`, `evaluator`); nothing is
implemented twice here.
"""
