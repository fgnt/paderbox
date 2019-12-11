import numpy as np
import soundfile

from paderbox.io import load_audio
from paderbox.io.audiowrite import dump_audio
from paderbox.testing.testfile_fetcher import get_file_path
    
import pytest

path = "tmp_audio.wav"


def get_audio_type(path):
    with soundfile.SoundFile(str(path), "r") as f:
        # 'PCM_16': np.int16,
        # 'FLOAT': np.float32,
        # 'DOUBLE': np.float64,
        return f.subtype


class TestIOAudio:

    # def test_default(self):
    #     a = np.array([1, 2, -4, 4], dtype=np.int16)
    #     dump_audio(a, path)
    #     assert get_audio_type(path) == 'PCM_16'
    #     b = load_audio(path)
    #     assert b.dtype == np.float64
    #     np.testing.assert_allclose(b * (2**15 / (2**15 - 1)), a / max(abs(a)))

    def test_default_wo_normalize(self):
        a = np.array([1, 2, -4, 4], dtype=np.int16)
        dump_audio(a, path, normalize=False)
        assert get_audio_type(path) == "PCM_16"
        b = load_audio(path)
        assert b.dtype == np.float64
        np.testing.assert_allclose(b, a / 2 ** 15)


    @pytest.mark.parametrize("array_dtype, dump_type, dumped_type, load_type, loaded_dtype",
        [
            (np.int16, np.int16, "PCM_16", np.int16, np.int16),
            (np.int32, np.int16, "PCM_16", np.int16, np.int16),
            (np.float32, np.int16, "PCM_16", np.int16, np.int16),
            (np.float64, np.int16, "PCM_16", np.int16, np.int16),
            (np.int16, np.int32, "PCM_32", np.int16, np.int16),
            (np.int32, np.int32, "PCM_32", np.int16, np.int16),
            (np.float32, np.int32, "PCM_32", np.int16, np.int16),
            (np.float64, np.int32, "PCM_32", np.int16, np.int16),
            (np.int16, np.float32, "FLOAT", np.int16, np.int16),
            (np.int32, np.float32, "FLOAT", np.int16, np.int16),
            (np.float32, np.float32, "FLOAT", np.int16, np.int16),
            (np.float64, np.float32, "FLOAT", np.int16, np.int16),
            (np.int16, np.float64, "DOUBLE", np.int16, np.int16),
            (np.int32, np.float64, "DOUBLE", np.int16, np.int16),
            (np.float32, np.float64, "DOUBLE", np.int16, np.int16),
            (np.float64, np.float64, "DOUBLE", np.int16, np.int16),
            (np.int16, None, "PCM_16", np.int16, np.int16),
            (np.int32, None, "PCM_32", np.int16, np.int16),
            (np.float32, None, "FLOAT", np.int16, np.int16),
            (np.float64, None, "DOUBLE", np.int16, np.int16),
            (np.int16, None, "PCM_16", np.int32, np.int32),
            (np.int32, None, "PCM_32", np.int32, np.int32),
            (np.float32, None, "FLOAT", np.int32, np.int32),
            (np.float64, None, "DOUBLE", np.int32, np.int32),
            (np.int16, None, "PCM_16", np.float32, np.float32),
            (np.int32, None, "PCM_32", np.float32, np.float32),
            (np.float32, None, "FLOAT", np.float32, np.float32),
            (np.float64, None, "DOUBLE", np.float32, np.float32),
            (np.int16, None, "PCM_16", np.float64, np.float64),
            (np.int32, None, "PCM_32", np.float64, np.float64),
            (np.float32, None, "FLOAT", np.float64, np.float64),
            (np.float64, None, "DOUBLE", np.float64, np.float64),
        ]
    )
    def test_dtype(
        self,
        array_dtype,
        dump_type,
        dumped_type,
        load_type,
        loaded_dtype,
    ):
        a = np.array([1, 2, -4, 4], dtype=array_dtype)
        dump_audio(a, path, dtype=dump_type, normalize=False)
        assert get_audio_type(path) == dumped_type
        b = load_audio(path, dtype=load_type)
        assert b.dtype == loaded_dtype

    @pytest.mark.parametrize("file,fails",
    [
        ('speech.sph', False),
        ('123_2alaw.sph', False),
        ('123_1pcle_shn.sph', True),
        ('123_1ulaw_shn.sph', True),
    ])
    def test_sph_files(self, file, fails):
        # Some SPHERE files can be read with soundfile, but not all.
        path = get_file_path(file)
        if fails:
            with pytest.raises(RuntimeError):
                load_audio(path)
        else:
            load_audio(path)
