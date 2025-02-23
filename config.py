DEBUG = False  # Set to False to disable debug messages

def debug_print(*args):
    if DEBUG:
        print("DEBUG: ", *args)
