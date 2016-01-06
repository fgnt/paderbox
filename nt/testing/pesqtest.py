'''
Christoph: remove pathes to your home folder !!!


# Program for testing PESQ functionality ( pesq.py)




import nt.evaluation.pesq as psq
import numpy
import sys
from scipy.io import wavfile

#Start


#Mode of PESQ operation. Default mode is nb- narrow band. For PESQ to run in wideband mode an additional parameter wb should be passed to the binary.
#pesqmode = input(" Enter the mode of operation for pesq - for narrow band enter 'nb' or for wide band enter 'wb' ")
pesqmode = 'nb'

# These are directories of reference audio,degraded audio and binary file of PESQ
reference = '/home/saju/matlab_toolbox/nt/speech_enhancement/test/speech.wav'
degraded = '/home/saju/matlab_toolbox/nt/speech_enhancement/test/speech_bab_0dB.wav'
dirbin = '/home/saju/matlab_toolbox/nt/speech_enhancement/pesq/build/bin/pesq'

rate, reference = wavfile.read(reference)

print(rate)
print(type(reference))
print(reference)
# call to the pesq function in class returns the pesq value which is then printed
obj = psq.pesq()
a,b = obj.pesqdef(reference,degraded,pesqmode,dirbin,rate)

# Print the PESQ scores::
    if pesqmode == 'nb':
        print('P.862 Prediction- Raw MOS ',b)
        print('P.862 Prediction- MOS-LQO ',a)
    elif pesqmode == 'wb':
        print('P.862 Prediction- MOS-LQO ',a)
    else:
        sys.exit()

'''


