import numpy as np
from nt.utils.matlab import Mlab

mlab = Mlab()


def dereverb(settings_file_path, x, stop_mlab=True):
    """
    This method wraps the matlab WPE-dereverbing-method. Give it the path to
    the settings.m and the wpe.p file and your reverbed signals as numpy matrix.
    Return value will be the dereverbed signals as numpy matrix.

    .. note:: The overall settings for this method are determined in the
        settings.m file. The wpe.p needs that settings.m file as input argument
        in order to work properly. Make sure that you read your audio signals
        accordingly.

    .. warning:: The settings file name MUST be 'wpe_settings'!

    :param settings_file_path: Path to wpe_settings.m and wpe.p
    :param x: NxC Numpy matrix of read audio signals. N denotes the signals'
        number of frames and C stands for the number of channels you provide
        for that signal
    :param stop_mlab: Whether matlab connection should be closed after execution
    :return: NxC Numpy matrix of dereverbed audio signals. N and C as above.
    """
    if not mlab.process.started:
        mlab.process.start()
    else:
        mlab.run_code('clear all;')

    settings = settings_file_path + "wpe_settings.m"

    # Check number of channels and set settings.m accordingly
    c = x.shape[1]
    modify_settings = False
    lines = []
    with open(settings) as infile:
        for line in infile:
            if 'num_mic = ' in line:
                if not str(c) in line:
                    line = 'num_mic = '+str(c)+";\n"
                    modify_settings = True
                else:
                    break   #ignore variable lines
            lines.append(line)
    if modify_settings:
        with open(settings, 'w') as outfile:
            for line in lines:
                outfile.write(line)

    #Process each utterance
    mlab.set_variable("x", x)
    mlab.set_variable("settings", settings)
    assert np.allclose(mlab.get_variable("x"), x)
    assert mlab.get_variable("settings") == settings
    mlab.run_code("addpath('"+settings_file_path+"');")

    # start wpe
    print("Dereverbing ...")
    mlab.run_code("y = wpe(x, settings);")
    # write dereverbed audio signals
    y = mlab.get_variable("y")

    if mlab.process.started and stop_mlab:
        mlab.process.stop()
    return y
