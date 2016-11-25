import numpy
import editdistance
import math

def sample_to_frame_idx(sample_idx, frame_size, frame_shift):
    """ Calculate corresponding frame index for sample index
        :param sample_idx: sample index
    """
    #start_offset = (frame_size - frame_shift)/2
    #frame_idx = (sample_idx - start_offset)//frame_shift
    #return max(0, frame_idx)
    ### To Match with the calculation at the input side (see stft(..))
    return math.ceil((sample_idx - frame_size + frame_shift)/frame_shift)


def argmax_ctc_decode(int_arr, label_handler):
    """ Decodes a ctc sequence

    :param int_arr: sequence to decode
    :param label_handler: label handler
    :type label_handler: CharLabelHandler
    :return: decoded sequence

    Example:
        >>> int_arr = numpy.random.randint()
        >>> argmax_ctc_decode()
    """

    max_decode = numpy.argmax(int_arr, axis=1)
    decode = numpy.zeros_like(max_decode)
    idx_dec = 0
    for idx, n in enumerate(max_decode):
        if idx > 0 and not n == max_decode[idx - 1]:
            decode[idx_dec] = n
            idx_dec += 1
        elif idx == 0:
            decode[idx_dec] = n
            idx_dec += 1
    idx_seq = [c for c in decode if c != 0]
    sequence = label_handler.ints2labels(idx_seq)
    return sequence


def argmax_ctc_decode_ler(dec_arr, ref_arr, label_handler):
    """ Decodes the ctc sequence and calculates label and word error rates

    :param dec_arr: ctc network output
    :param ref_arr: reference sequence (as int array)
    :param label_handler: label handler
    :return: decode, ler, wer
    """
    dec_seq = argmax_ctc_decode(dec_arr, label_handler)
    ref_seq = label_handler.ints2labels(ref_arr)
    ler = editdistance.eval(list(dec_seq), list(ref_seq)) / len(list(ref_seq))
    return dec_seq, ler


def argmax_ctc_decode_ler_wer(dec_arr, ref_arr, label_handler):
    """ Decodes the ctc sequence and calculates label and word error rates

    :param dec_arr: ctc network output
    :param ref_arr: reference sequence (as int array)
    :param label_handler: label handler
    :return: decode, ler, wer
    """
    dec_seq = argmax_ctc_decode(dec_arr, label_handler)
    ref_seq = label_handler.ints2labels(ref_arr)
    ler = editdistance.eval(list(dec_seq), list(ref_seq)) / len(list(ref_seq))
    wer = editdistance.eval(dec_seq.split(), ref_seq.split()) \
          / len(ref_seq.split())
    return dec_seq, ler, wer


def argmax_ctc_decode_with_stats(dec_arr, ref_arr, label_handler,
                                 include_space=False):
    """ Decodes the ctc sequence, calculates label and word error rates and
    returns various stats

    :param dec_arr: ctc network output
    :param ref_arr: reference sequence (as int array)
    :param label_handler: label handler
    :param include_space: The network can output a space. Thus we can also
        calculate word statistics. Otherwise word statistics will be 0/-1
    :return: decode, ler, wer, label_errors, word_errors, labels, words
    """
    dec_seq = argmax_ctc_decode(dec_arr, label_handler)
    ref_seq = label_handler.ints2labels(ref_arr)
    if include_space:
        ref_words = ''.join(ref_seq).split()
        dec_words = ''.join(dec_seq).split()
        word_errors = editdistance.eval(dec_words, ref_words)
        wer = word_errors / len(ref_words)
    else:
        word_errors = -1
        wer = -1
        ref_words = []
    label_errors = editdistance.eval(dec_seq, ref_seq)
    ler = label_errors / len(ref_seq)
    return dec_seq, ler, wer, label_errors, word_errors, \
           len(ref_seq), len(ref_words)
