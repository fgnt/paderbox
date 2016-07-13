import pandas as pd
from bson.objectid import ObjectId


def filter_columns(frame, entries):
    """

    :param frame:
    :param entries:
    :return:
    """

    # TODO: Unclear, why this works.
    def _in(lv, l):
        """checks for every level name in l if this level name is in level lv
        """
        res = None
        for k in l:
            if res is None:
                res = (lv == k)
            else:
                res |= (lv == k)
        return res

    mask = None
    lv0 = frame.columns.get_level_values(0)  # get column names for level 0
    lv1 = frame.columns.get_level_values(1)  # get column names for level 1

    # for every column name of level 0, check if in list entries. If so, check
    # if all values are needed (entries[k] = True) or to filter in level names
    # of level 1
    for k in entries.keys():
        v = entries[k]
        if type(v) is list:
            new_mask = (lv0 == k) & (_in(lv1, v))
        else:
            # take columns with first level name = entries[k]
            new_mask = (lv0 == k)

        # concatenate masks
        mask = new_mask if mask is None else mask | new_mask

    return frame.loc[:, mask]  # apply mask


def print_columns(df, indent=0):
    # TODO: This does not work reliably yet
    try:
        for column in df.columns.levels[0]:
            print('    ' * indent + column)
            if not isinstance(df[column], pd.Series):
                print_columns(df[column], indent=indent + 1)
    except AttributeError:
        for column in df.columns:
            print('    ' * indent + column)


def make_css_mark(mask, color_str='#FFFFFF'):
    css = ''
    for i in range(len(mask)):
        if mask.iloc[i]:
            css += 'tbody tr:nth-child(%d) {background-color: %s}\n' % (
                i + 1, color_str
            )
    return css


def colorize_and_display_dataframe(df, column='status', color_dict=dict(
    COMPLETED='#84BD00',
    FAILED='#FFC600',
    INTERRUPTED='#FF8200',
    RUNNING='#009FDF'
)):
    """ Uses UPB colors to colorize table.

    Generates a unique class name to not overwrite existing CSS code and then
    colorizes rows based on experiment status.

    Args:
        df: Filtered data frame i.e. from `filter_columns()`.

    Returns: None

    """
    import uuid
    from IPython.display import HTML

    random_class_name = 'tab{}'.format(uuid.uuid1())

    html = ''

    html += '<style type=text/css>\n'
    # html += '.tab_' + str(uuid.uuid1()) + '\n'
    html += '.{}\n'.format(random_class_name)

    for key, value in color_dict.items():
        html += make_css_mark((df[column] == key), color_str=value)

    html += 'thead tr {background-color: #C7C9C7}\n'
    html += '</style>\n'

    html += df.to_html(escape=False, classes=random_class_name)

    return HTML(html)


def set_values(runs, _id, values):
    """
    Sets all key-value paris found in values in the database for the entry with
    id _id. If a specified value already exists, it gets replaced.
    :param runs:
    :param _id:
    :param values: dict containing as key value pairs: key: key of the value to
        be added or replaced, value: value to be set
    :return:
    """
    if not isinstance(_id, ObjectId):
        _id = ObjectId(_id)

    to_replace = runs.find_one({'_id': _id})

    for key in values.keys():
        value = values[key]
        to_replace[key] = value

    runs.replace_one({'_id': _id}, to_replace)
