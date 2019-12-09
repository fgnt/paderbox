import numpy as np
from IPython.display import HTML, Audio, display_html, Image
import shutil
# from paderbox.database.keys import *
from paderbox.io.audioread import load_audio, audio_length, is_nist_sphere_file
import pathlib

from paderbox.transform import spectrogram
from paderbox.visualization import figure_context


VALID_AUDIO_EXTENSIONS = ('wav', 'flac', 'ogg')


# ToDo: delete this code in public repo


class Templates:
    ul = '<ul {args}>{content}</ul>'
    li = '<li>{content}</li>'
    card = '''
        <div class="card">
            <div class="cardheader">
                <h2>{header}</h2>
                <div>(dataset size: {size})</div>
                <div>(example: {example_key})</div>
            </div>
            <div class="cardcontents">
                {content}
            </div>
        </div>
    '''
    warning = '<span class="warning" >{content}</span>'
    horizontal_divided_cell = '''
        <div class="flex_horizontal">
            <div width:550px>{left}</div><div style="margin-left:40px">
            {right}</div>
        </div>'''
    error = '<div class="error">{content}</div>'
    style = '''
    <style>
    .error {
        color: red;
        align: left;
        text-align: left;
        border-style: solid;
        display: inline-block;
    }

    .warning {
        color: orange;
    }

    .card {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        display: inline-block;
        margin: 10px;
        vertical-align: top;
        transition: 0.3s;
    }

    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }

    .cardheader {
        align: center;
        background: #eeeeee;
    }

    .cardcontents {
        padding: 10px;
    }

    .small {
        max-width: 900px;
    }

    .flex_horizontal {
        display: flex;
        flex-direction: row;
    }
    </style>
    '''


def audio_to_html(data_or_str, embed=False, max_audio_length=20):
    html = ''
    if not is_audio(data_or_str):
        raise ValueError(f'Unknown audio format: {data_or_str}')

    if embed:
        if is_audio_path(data_or_str):
            audio_data, sample_rate = load_audio(data_or_str,
                                                 return_sample_rate=True)
        else:
            assert is_audio_array(data_or_str)
            audio_data = data_or_str
            sample_rate = 16000     # assume sampling rate of 16kHz

        if max_audio_length is not None and \
            max_audio_length * sample_rate <= audio_data.shape[0]:
            html = f'(original length was ' \
                    f'{int(audio_data.shape[0]/sample_rate)}s)'
            audio_data = audio_data[:max_audio_length * sample_rate]

        html += Audio(audio_data, rate=sample_rate)._repr_html_()
    else:
        assert is_audio_path(data_or_str)
        if is_nist_sphere_file(data_or_str):
            raise ValueError(
                f'Audio at {data_or_str} is in nist/sphere format, '
                'which the ipython Audio applet cannot play. '
                'Use embed_audio=True instead.')
        path, length = cache_audio_local(data_or_str, max_audio_length)
        if path is None:
            html = Templates.warning.format(
                content=f'Audio too long to display ({length} seconds)')
        else:
            html = Audio(filename=str(path))._repr_html_()
    return html


def plot_to_html(data_or_str, image_width=None, max_audio_length=20):
    dst_path = pathlib.Path('images')
    if isinstance(data_or_str, str):
        dst_path /= (str(pathlib.Path(data_or_str)) + '.png')[1:]
        audio_data, sample_rate = load_audio(data_or_str,
                                              return_sample_rate=True)
    else:
        # use a random number to hopefully get distinguishable file names
        dst_path /= str(np.random.randint(0)) + '.png'
        audio_data, sample_rate = data_or_str

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if audio_data.shape[0] / sample_rate <= max_audio_length:

            from paderbox.visualization.plot import spectrogram as plot_spectrogram
            from matplotlib import pyplot as plt

            to_plot = spectrogram(audio_data)
            with figure_context():
                plot_spectrogram(to_plot)
                plt.savefig(str(dst_path), pad_inches=0, bbox_inches='tight')
                plt.close()

    if dst_path.exists():
        return Image(url=str(dst_path), width=image_width)._repr_html_()
    else:
        return None


def cache_audio_local(path, max_audio_length, cache_dir=pathlib.Path('./audio')):
    """
    Copies the file specified by `path` to cache_dir in the same folder
    structure that is present at the source. The file should be in an audio
    format readable by `nt.io.audioread.load_audio`.

    :param path: Audio file to cache locally
    :param max_audio_length: maximum length of files to be cached in seconds.
        Files that exceed this limit are not copied. The returned path is then
         None.
    :param cache_dir: directory to store the cached files in
    :return: Tuple: (Path where the cached file is located or None,
                     length of audio in seconds)
    """
    local_path = cache_dir / str(path)[1:]
    local_path.parent.mkdir(exist_ok=True, parents=True)
    length = audio_length(path, unit='seconds')
    if max_audio_length is not None and length > max_audio_length:
        return None, length
    if not local_path.exists():
        shutil.copy(path, local_path)
    return local_path, length


def create_from_dict(d, embed_audio=False, max_audio_length=20, depth=0,
                     image_width=None):
    html = ''
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            html += Templates.li.format(
                content=f'{k}: ' + example_to_html(v, max_audio_length,
                                                   embed_audio, depth+1,
                                                   image_width))
        elif is_audio(v):
            html += Templates.li.format(
                content=f'{k}: ' + audio_to_html(v, embed_audio,
                                                 max_audio_length)
            )
        elif isinstance(v, (str, int, float)):
            html += Templates.li.format(content=f'{k}: {v}')
        else:
            raise TypeError(f'Unexpected type {type(v)} of element {v}.')

    to_image = list(d.items())[0]
    image = plot_to_html(to_image[1], image_width, max_audio_length)
    html = Templates.horizontal_divided_cell.format(
        left=Templates.ul.format(args=f'id=level{depth}', content=html),
        right='' if image is None else
                f'<h3>Spectrogram of channel {to_image[0]}</h3>' + image)
    return html


def is_audio_path(v):
    return isinstance(v, str) and any(v.endswith(ext) for ext in
                                      VALID_AUDIO_EXTENSIONS)


def is_audio_array(v):
    return isinstance(v, np.ndarray) and v.shape[0] <= 2


def is_audio(v):
    return is_audio_path(v) or is_audio_array(v)


def is_dict_of_audio(d):
    first_key = list(d.keys())[0]
    v = d[first_key]
    return is_audio(v)


def example_to_html(audio_dict, max_audio_length=None, embed_audio=False,
                        depth=0, image_width=None):
    """
    Creates an html representation of a single example by recursively walking
    through the nested dict structure.

    If `embed_audio` is true, all numpy arrays are assumed to be audio data and
    embedded in the html code. If `embed_audio` is false, all Strings that end
    with a valid audio extension are cached locally in a folder besides the
    notebook and a link to that file is included in the html.

    :param audio_dict: Example dict
    :param max_audio_length: maximum length of audio files that are displayed.
        If audio is embedded, longer audio snippets are truncated to this
        length. If audio is not embedded, files that exceed `max_audio_legnth`
        will not be displayed.
    :param embed_audio:
    :return: html code as string
    """
    html = ''
    if isinstance(audio_dict, list):
        audio_dict = {i: v for i, v in enumerate(audio_dict)}

    if isinstance(audio_dict, dict):
        if is_dict_of_audio(audio_dict):
            html += create_from_dict(audio_dict, embed_audio=embed_audio,
                                     max_audio_length=max_audio_length,
                                     depth=depth+1, image_width=image_width)
        else:
            tmp = ''
            for k, v in audio_dict.items():
                if not (embed_audio and k == AUDIO_PATH):
                    tmp += Templates.li.format(
                        content=f'{k}: ' +
                                example_to_html(v, max_audio_length,
                                                embed_audio, depth+1,
                                                image_width=image_width))
            html += Templates.ul.format(args=f'id=level{depth}', content=tmp)
    elif is_audio(audio_dict):
        html += Templates.horizontal_divided_cell.format(
            left=audio_to_html(audio_dict, embed_audio, max_audio_length),
            right=plot_to_html(audio_dict, image_width, max_audio_length)
        )
    else:
        html += str(audio_dict)
    return html


def dataset_to_html_card(dataset_key, dataset, embed_audio, max_audio_length,
                         image_width):
    example_key = ''
    if isinstance(dataset, list):
        content = Templates.error.format(
            content=f'Dataset "{dataset_key}" still uses old format!')
    else:
        example_key = list(dataset.keys())[0]
        example = dataset[example_key]
        if AUDIO_PATH not in example:
            content = Templates.error.format(
                content='ERROR: No Audio Paths found!')
        else:
            try:
                content = example_to_html(
                    example,
                    max_audio_length=max_audio_length,
                    embed_audio=embed_audio, image_width=image_width)
            except Exception as e:
                content = Templates.error.format(
                    content=f'ERROR: {type(e)}: {e}')
    return Templates.card.format(header=dataset_key, example_key=example_key,
                                 content=content, size=len(dataset))


def database_to_html(database_dict, embed_audio,
                     max_audio_length=20, image_width=None,
                     datasets=None):
    try:
        html = ''

        # read single example for each dataset
        for dataset_key, dataset in database_dict[DATASETS].items():
            if datasets is None or dataset_key in datasets:
                html += dataset_to_html_card(dataset_key, dataset,
                                             embed_audio=embed_audio,
                                             max_audio_length=max_audio_length,
                                             image_width=image_width)

    except Exception as e:
        html = Templates.error.format(content=f'ERROR: {type(e)}: {e}')
    return html


def display_database_html(db, embed_audio=False, max_audio_length=20,
                          image_width=200, datasets=None):
    """
    Creates an example html representation for this database and displays it
    using `IPython.display.display_html`. Creates an html
    card for each dataset containing one example of the dataset. All Audio files
    referenced in the example are included as playable audio inside of the html.

    :param db: database to create the example representation for
    :param embed_audio: whether to embed the audio data in the html code
            (produces huge html files that may exceed the notebook limit) or
            to copy the referenced files into a directory accessible by the
            notebook server and include a link.
    :param max_audio_length: Maximum length of audio files to be displayed.
            Audio files that exceed this limit are either truncated
            (if `embed_audio` is True) or not displayed (if `embed_audio` is
            False).
    :param datasets:
    """
    display_html(HTML(Templates.style + database_to_html(
        db.database_dict, embed_audio=embed_audio,
        max_audio_length=max_audio_length, image_width=image_width,
        datasets=datasets
    )))
