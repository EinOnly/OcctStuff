from classA import A
from classB import B
from context import context

def init():
    ctx = context()
    a_instance = A(ctx)
    b_instance = B(ctx)
    return a_instance, b_instance

def main():
    a, b = init()
    ra = a._run("start")
    rb = b._run(ra)

if __name__ == "__main__":
    main()