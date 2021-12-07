from paderbox.io.cache import url_to_local_path
from paderbox.utils.deprecation import deprecated

fetch_file_from_url = deprecated('function has been moved to io.cache as url_to_local_path')(url_to_local_path)


def get_file_path(file_name):
    """
    Looks up path to a test audio file and returns to the local file.

    Args:
        file_name: audio file needed for the test

    Returns: Path to audio test file

    """
    _pesq = "https://github.com/ludlows/python-pesq/raw/master/audio/"
    _pb_bss = "https://github.com/fgnt/pb_test_data/raw/master/bss_data/" \
              "low_reverberation/"

    url_ = {
        'sample.wav': _pb_bss + "speech_source_0.wav",

        'observation.wav': _pb_bss+"observation.wav",  # multi channel
        'speech_source_0.wav': _pb_bss+"speech_source_0.wav",
        'speech_source_1.wav': _pb_bss+"speech_source_1.wav",
        'speech_image_0.wav': _pb_bss+"speech_image_0.wav",  # multi channel
        'speech_image_1.wav': _pb_bss+"speech_image_1.wav",  # multi channel
        'noise_image.wav': _pb_bss+"noise_image.wav",  # multi channel

        'speech.wav': _pesq + "speech.wav",
        "speech_bab_0dB.wav": _pesq + "speech_bab_0dB.wav",

        # pylint: disable=line-too-long
        # Found on https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p04.html
        'speech.sph': 'https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/data/speech.sph',
        '123_1pcbe_shn.sph': 'https://github.com/robd003/sph2pipe/raw/master/test/123_1pcbe_shn.sph',
        '123_1pcle_shn.sph': 'https://github.com/robd003/sph2pipe/raw/master/test/123_1pcle_shn.sph',
        '123_1ulaw_shn.sph': 'https://github.com/robd003/sph2pipe/raw/master/test/123_1ulaw_shn.sph',
        '123_2alaw.sph': 'https://github.com/robd003/sph2pipe/raw/master/test/123_2alaw.sph',
    }[file_name]

    return url_to_local_path(url_, file_name)
