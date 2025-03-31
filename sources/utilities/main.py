from log import CORELOG
from interface import ui_main
def main():
    CORELOG.info("Hello")
    CORELOG.warn("Hello")
    CORELOG.erro("Hello")
    ui_main()
    CORELOG.sync()

if __name__ == "__main__":
    main()

'''

'''