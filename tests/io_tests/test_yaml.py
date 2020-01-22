import tempfile
from pathlib import Path
import paderbox as pb


def test_yaml_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path = (tmpdir / 'test.yaml')
        path.write_text('a: 1\nb: 2')

        assert pb.io.load_yaml(path) == {'a': 1, 'b': 2}
        assert pb.io.load_yaml_unsafe(path) == {'a': 1, 'b': 2}


def test_yaml_dump():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path = (tmpdir / 'test.yaml')

        data = {'a': 1, 'b': 2}

        pb.io.dump_yaml(data, path)
        assert path.read_text() == 'a: 1\nb: 2\n'

        pb.io.dump_yaml_unsafe(data, path)
        assert path.read_text() == 'a: 1\nb: 2\n'
