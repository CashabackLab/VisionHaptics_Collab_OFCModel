import shelve

# save session
def save_workspace(workspace, globals, save_directory, save_session_name):
    bk = shelve.open(save_directory + '\{}.pkl'.format(save_session_name), 'n')
    for k in workspace:
        print(k)
        try:
            bk[k] = globals[k]
        except Exception:
            pass
    bk.close()

# to load your session
def return_workspace(saved_directory, saved_session_name):
    bk_restore = shelve.open(saved_directory + '\{}.pkl'.format(saved_session_name))
    loaded_workspace = {}
    for k in bk_restore:
        print(k)
        try:
            loaded_workspace[k] = bk_restore[k]
        except:
            pass
    bk_restore.close()

    return loaded_workspace