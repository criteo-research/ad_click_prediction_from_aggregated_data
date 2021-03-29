# Avoid printing stacktrace on "KeyboardInterrupt"
import sys

ipython = get_ipython()
try:
    _showtraceback_default
except:
    _showtraceback_default = ipython._showtraceback


def exception_handler(exception_type, exception, traceback):
    if exception_type is KeyboardInterrupt:
        print("KeyboardInterrupt", file=sys.stderr)
    else:
        # print(str(exception_type))
        return _showtraceback_default(exception_type, exception, traceback)


ipython._showtraceback = exception_handler


def logline(s):
    print(s)
    with open("log.txt", "a") as file:
        file.write(s + "\n")
