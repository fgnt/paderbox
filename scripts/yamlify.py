import argparse
import yaml
from pathlib import Path
import paderbox as pb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human readable JSON file.')
    parser.add_argument(
        'path',
        type=str,
        help='Path to JSON. If path is a directory, guess JSON name.',
    )
    parse_result = parser.parse_args()
    path = Path(parse_result.path)

    if path.is_dir():
        candidates = 'init.json config.json'.split()
        for candidate in candidates:
            candidate_path = path / candidate
            if candidate_path.is_file():
                path = candidate_path
                break

    print(yaml.dump(pb.io.load_json(path)))
