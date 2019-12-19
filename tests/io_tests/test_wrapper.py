import tempfile
import pytest
from pathlib import Path
import numpy as np
import paderbox as pb


@pytest.mark.parametrize(
    'file_name,load_fails,load_unsafe',
    [
        pytest.param(
            file_name, load_fails, load_unsafe, **kwargs,
            id=f'{file_name.replace("file.", "").replace(".", "_")}_load_fails_{load_fails}_load_unsafe_{load_unsafe}'
        )
        for file_name, load_fails, load_unsafe, kwargs in [
            ('file.json', False, False, {}),
            ('file.yaml', False, False, {}),
            ('file.pkl', True, False, {}),
            ('file.pkl', False, True, {}),
            ('file.dill', True, False, {}),
            ('file.dill', False, True, {}),
            ('file.wav', False, False, {}),
            ('file.mat', False, False, {}),
            ('file.npy', True, False, {}),
            ('file.npy', False, True, {}),
            ('file.npz', False, False, {}),
            ('file.pth', True, False, {'marks': pytest.mark.torch}),
            ('file.pth', False, True, {'marks': pytest.mark.torch}),
            ('file.json.gz', False, False, {}),
            ('file.pkl.gz', True, False, {}),
            ('file.pkl.gz', False, True, {}),
            ('file.npy.gz', True, False, {}),
            ('file.npy.gz', False, True, {}),
        ]
    ]
)
def test_load_dump(file_name, load_fails, load_unsafe):
    with tempfile.TemporaryDirectory() as tmpdir:
        obj = [1, 2, 4]
        if file_name == 'file.mat':
            obj = {'array': obj}

        path = Path(tmpdir) / file_name
        # Unsafe dump is ok, no security problems
        pb.io.dump(obj, path, unsafe=True)

        assert not (load_fails and load_unsafe), 'Only one can be True'
        if load_fails:
            with pytest.raises(AssertionError):
                pb.io.load(path)
        else:
            if file_name == 'file.wav':
                correction = (2 ** 15 - 1) / (2 ** 15)
                desired = correction * np.array(obj) / np.amax(obj)
                np.testing.assert_allclose(pb.io.load(path, unsafe=load_unsafe), desired, rtol=1e-4)
            elif file_name == 'file.mat':
                assert np.all(pb.io.load(path, unsafe=load_unsafe)['array'] == obj['array']), path
            elif file_name in ['file.npz', 'file.npz.gz']:
                assert np.all(pb.io.load(path, unsafe=load_unsafe)['arr_0'] == obj), path
            else:
                assert np.all(pb.io.load(path, unsafe=load_unsafe) == obj), path
