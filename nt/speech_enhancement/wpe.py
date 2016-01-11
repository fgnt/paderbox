import numpy as np
from nt.utils.matlab import Mlab
from nt.io import audioread, audiowrite


def dereverb(settings_file_path=
             '/net/home/wilhelmk/PythonToolbox/'
             'nt/speech_enhancement/utils/wpe_settings.m',
             input_file_paths=
             {'/net/home/wilhelmk/PythonToolbox/'
             'nt/speech_enhancement/utils/sample_ch1.wav':1,},
             output_dir_path=
             '/net/home/wilhelmk/PythonToolbox/'
             'nt/speech_enhancement/utils/',
             sample_rate=16000
             ):

    mlab = Mlab()
    #Process each utterance
    file_no = 0
    for utt, num_channels in input_file_paths.items():
        file_no += 1
        print("Processing file no. {0} ({1} file(s) to process in total)"
              .format(file_no, len(input_file_paths)))
        noisy_audiosignals = np.ndarray(
            #shape=[audioread.getparams(utt).nframes, num_channels]
             shape=[sample_rate*1, num_channels]
             )
        #print(str(audioread.getparams(utt).nframes)+" frames per channel")
        for cha in range(num_channels):
            #read microphone signals (each channel)
            print(" - Reading channel "+str(cha+1))
            utt_to_read = utt.replace('ch1', 'ch'+str(cha+1))
            signal = audioread.audioread(path= utt_to_read,
                                         sample_rate=sample_rate,
                                         duration=1)
            if not noisy_audiosignals.shape[0] == len(signal):
                raise Exception("Signal "+utt_to_read+" has a different size "
                                                      "than other signals.")
            else:
                noisy_audiosignals[:,cha] =  signal

        mlab.set_variable("x",noisy_audiosignals)
        mlab.set_variable("settings",settings_file_path)
        assert np.allclose(mlab.get_variable("x"), noisy_audiosignals)
        assert mlab.get_variable("settings")== settings_file_path
        mlab.run_code_print("addpath('"+output_dir_path+"')")
        # start wpe
        print("Dereverbing ...")
        mlab.run_code_print("y = wpe(x, settings)")
        # write dereverbed audio signals
        y = mlab.get_variable("y")
        for cha in range(num_channels):
            utt_to_write = utt.replace('ch1', 'ch'+str(cha+1)+'_derev')
            print(" - Writing channel "+str(cha+1))
            audiowrite.audiowrite(y[:,cha],
                                  output_dir_path + utt_to_write,
                                  sample_rate )
    print("Finished successfully.")

if __name__ == "__main__":
    from nt.speech_enhancement import wpe

    wpe.dereverb()