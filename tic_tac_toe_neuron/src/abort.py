import multiprocessing

# Abort file name
# Presence of this file in working directory can be used by various parallel processes to abort their work
abort_file = "abort"

# Pause file name
# Presence of this file in working directory can be used by various parallel processes to pause their work until the file is removed
pause_file = "pause"

# Call to abort execution of all processes
def abort():
    with open(abort_file, "w") as F:
        F.write("")

# Call to pause execution of all processes
def pause():
    with open(pause_file, "w") as F:
        F.write("")

# Call to resume execution of all processes
def resume():
    if (os.path.exists(pause_file)):
        os.remove(pause_file)

# The multiprocessing library is not compatible with decorated functions,
# so this is a workaround solution
def funcAbortWrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Generator {multiprocessing.current_process().name} raised an error '{e}'. Aborting now.", flush=True)
        abort()
        raise
