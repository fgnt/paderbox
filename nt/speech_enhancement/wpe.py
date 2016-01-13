import numpy as np
from nt.utils.matlab import Mlab



def dereverb(settings_file_path, x):
    """
    This method wraps the matlab WPE-dereverbing-method. Give it the path to
    the settings.m and the wpe.p file and your reverbed signals as numpy matrix.
    Return value will be the dereverbed signals as numpy matrix.

    .. note:: The overall settings for this method are determined in the
        settings.m file. The wpe.p needs that settings.m file as input argument
        in order to work properly. Make sure that you read your audio signals
        accordingly,

    :param settings_file_path: Path to settings.m and wpe.p
    :param x: NxC Numpy matrix of read audio signals. N denotes the signals'
        number of frames and C stands for the number of channels you provide
        for that signal
    :return: NxC Numpy matrix of dereverbed audio signals. N and C as above.
    """


    # todo: Notebook schreiben, das ein Audiosignal liest,es verhallt mit einer
    #  selbst erstellten RIR, dann enthallt mit wpe und beide audiosignale ausgeben kann
    mlab = Mlab()

    #Process each utterance
    mlab.set_variable("x",x)
    mlab.set_variable("settings",settings_file_path+"wpe_settings.m")
    assert np.allclose(mlab.get_variable("x"), x)
    assert mlab.get_variable("settings") == settings_file_path+"wpe_settings.m"
    mlab.run_code_print("addpath('"+settings_file_path+"');")

    # start wpe
    print("Dereverbing ...")
    mlab.run_code_print("y = wpe(x, settings);")
    # write dereverbed audio signals
    y = mlab.get_variable("y")
    return y
