import matlab.engine


def run_matlab_in_python():
    eng = matlab.engine.start_matlab()
    eng.writing(nargout=0)
    eng.quit()


run_matlab_in_python()
