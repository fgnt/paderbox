import pandas as pd
import six
from bson.objectid import ObjectId
try:
    # Version lower than 0.18
    from pandas.core.format import HTMLFormatter, DataFrameFormatter, \
        _get_level_lengths
except ImportError:
    # versions higher than 0.18
    from pandas.formats.format import HTMLFormatter, DataFrameFormatter, \
        _get_level_lengths

import pandas.core.common as com
from nt.utils.misc import update_dict as _update_dict


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


def colorize_and_display_dataframe(df, column='status', color_dict=None):
    """ Uses UPB colors to colorize table.

    Generates a unique class name to not overwrite existing CSS code and then
    colorizes rows based on experiment status.

    Args:
        df: Filtered data frame i.e. from `filter_columns()`.

    Returns: None

    """
    import uuid
    from IPython.display import HTML

    if color_dict is None:
        color_dict = dict(
            COMPLETED='#84BD00',
            FAILED='#FFC600',
            INTERRUPTED='#FF8200',
            RUNNING='#009FDF'
        )

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


def set_values(runs, _id, update_dict):
    """
    Sets all key-value pairs found in values in the database for the entry with
    id _id. If a specified value already exists, it gets replaced.
    :param runs:
    :param _id:
    :param update_dict: dict containing as key value pairs: key: key of the
        value to be added or replaced, value: value to be set
    :return:

    Example call:

    .. code::
        sacred_manager.set_db_values(
            '57727e68d4165e2625aa26e9',
            {
                'status': 'INTERRUPTED',
                'config': {
                    'comment': 'fun'
                }
            }
        )
    """
    if not isinstance(_id, ObjectId):
        _id = ObjectId(_id)

    to_replace = runs.find_one({'_id': _id})
    to_replace = _update_dict(to_replace, update_dict)
    runs.replace_one({'_id': _id}, to_replace)


class FormattedHeaderHTMLFormatter(HTMLFormatter):
    def format_header_cell(self, header_text, level_number):
        """
        This has to be overwritten by subclasses that want to apply a format to
        the header.
        :param header_text: content as text of the header_field
        :param level_number: if using a MultiIndex, this is the levenumber
        :return: tuple (new_content_of_header_cell_which_may_contain_html,
            tags to be added to the <th/> tag as one string)
        """
        return header_text, ''

    def _write_header(self, indent):
        truncate_h = self.fmt.truncate_h
        row_levels = self.frame.index.nlevels
        if not self.fmt.header:
            # write nothing
            return indent

        def _column_header():
            if self.fmt.index:
                row = [''] * (self.frame.index.nlevels - 1)
            else:
                row = []

            if isinstance(self.columns, pd.MultiIndex):
                if self.fmt.has_column_names and self.fmt.index:
                    row.append(single_column_table(self.columns.names))
                else:
                    row.append('')
                style = "text-align: %s;" % self.fmt.justify
                row.extend([single_column_table(c, self.fmt.justify, style) for
                            c in self.columns])
            else:
                if self.fmt.index:
                    row.append(self.columns.name or '')
                row.extend(self.columns)
            return row

        self.write('<thead>', indent)
        row = []

        indent += self.indent_delta

        if isinstance(self.columns, pd.MultiIndex):
            template = 'colspan="%d" halign="left"'

            if self.fmt.sparsify:
                # GH3547
                sentinel = com.sentinel_factory()
            else:
                sentinel = None
            levels = self.columns.format(sparsify=sentinel,
                                         adjoin=False, names=False)
            level_lengths = _get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            for lnum, (records, values) in enumerate(zip(level_lengths,
                                                         levels)):
                if truncate_h:
                    # modify the header lines
                    ins_col = self.fmt.tr_col_num
                    if self.fmt.sparsify:
                        recs_new = {}
                        # Increment tags after ... col.
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            elif tag + span > ins_col:
                                recs_new[tag] = span + 1
                                if lnum == inner_lvl:
                                    values = values[:ins_col] + (u'...',) + \
                                             values[ins_col:]
                                else:  # sparse col headers do not receive a ...
                                    values = (
                                        values[:ins_col] + (
                                            values[ins_col - 1],) +
                                        values[ins_col:])
                            else:
                                recs_new[tag] = span
                            # if ins_col lies between tags, all col headers get ...
                            if tag + span == ins_col:
                                recs_new[ins_col] = 1
                                values = values[:ins_col] + (u('...'),) + \
                                         values[ins_col:]
                        records = recs_new
                        inner_lvl = len(level_lengths) - 1
                        if lnum == inner_lvl:
                            records[ins_col] = 1
                    else:
                        recs_new = {}
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            else:
                                recs_new[tag] = span
                        recs_new[ins_col] = 1
                        records = recs_new
                        values = values[:ins_col] + [u('...')] + values[
                                                                 ins_col:]

                name = self.columns.names[lnum]
                row = [''] * (row_levels - 1) + ['' if name is None
                                                 else com.pprint_thing(name)]

                if row == [""] and self.fmt.index is False:
                    row = []

                tags = {}
                j = len(row)
                for i, v in enumerate(values):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template % records[i]
                    else:
                        continue

                    # format headers
                    v, tag = self.format_header_cell(v, lnum)
                    row.append(v)
                    if j not in tags.keys():
                        tags[j] = tag
                    else:
                        tags[j] += ' ' + tag

                    j += 1
                self.write_tr(row, indent, self.indent_delta, tags=tags,
                              header=True)
        else:
            col_row = _column_header()
            align = self.fmt.justify

            if truncate_h:
                ins_col = row_levels + self.fmt.tr_col_num
                col_row.insert(ins_col, '...')

            # format headers
            col_row = [self.format_header_cell(v, 0) for v in col_row]
            tags = {i: col_row[i][1] for i in range(len(col_row))}
            col_row = [v[0] for v in col_row]

            self.write_tr(col_row, indent, self.indent_delta, header=True,
                          align=align, tags=tags)

        if self.fmt.has_index_names and self.fmt.index:
            row = [
                      x if x is not None else '' for x in self.frame.index.names
                      ] + [''] * min(len(self.columns), self.max_cols)
            if truncate_h:
                ins_col = row_levels + self.fmt.tr_col_num
                row.insert(ins_col, '')
            self.write_tr(row, indent, self.indent_delta, header=True)

        indent -= self.indent_delta
        self.write('</thead>', indent)

        return indent


class RotatedTHeadHTMLFormatter(FormattedHeaderHTMLFormatter):
    def __init__(self, formatter, classes=None, max_rows=None, max_cols=None,
                 notebook=False, min_rotation_level=1):
        super().__init__(formatter, classes, max_rows, max_cols, notebook)
        self.min_rotation_level = min_rotation_level

    def format_header_cell(self, header_text, level_number):
        if level_number >= self.min_rotation_level:
            return '<span class="intact">' + header_text + '</span>', \
                   'class="rotate"'
        else:
            return header_text, ''


def format_html(formatter_class, df, css_style_string='', buf=None,
                columns=None,
                col_space=None,
                header=True, index=True, na_rep='NaN', formatters=None,
                float_format=None, sparsify=None, index_names=True,
                justify=None, bold_rows=True, classes=None,
                max_rows=None, max_cols=None, show_dimensions=False,
                notebook=False, **kwargs):
    """
    Returns a string containing the html-formatted DataFrame df styled with the
    css style provided in css_style_string. Uses the HTMLFormatter class
    formatter_class. If you want to style the whole table, you have to specify
    the css class to use in the parameter classes.

    For more information on the parameters see the pandas documentation for
    DataFrameFormatter (currently empty for this topic...)

    :param df: The DataFrame to be formatted
    :param **kwargs: These arguments are passed as they are to the constructor
        of an instance of formatter_class to be able to pass arguments to
        custom formatters.
    :param css_style_string: CSS-Style to be embedded in the html document.
    """
    formatter = DataFrameFormatter(df,
                                   buf=buf, columns=columns,
                                   col_space=col_space, na_rep=na_rep,
                                   formatters=formatters,
                                   float_format=float_format,
                                   sparsify=sparsify,
                                   justify=justify,
                                   index_names=index_names,
                                   header=header, index=index,
                                   bold_rows=bold_rows,
                                   escape=False,
                                   max_rows=max_rows,
                                   max_cols=max_cols,
                                   show_dimensions=show_dimensions)

    html_renderer = formatter_class(formatter, classes=classes,
                                    max_rows=formatter.max_rows,
                                    max_cols=formatter.max_cols,
                                    notebook=notebook, **kwargs)
    if hasattr(formatter.buf, 'write'):
        html_renderer.write_result(formatter.buf)
    elif isinstance(formatter.buf, six.string_types):
        with open(formatter.buf, 'w') as f:
            html_renderer.write_result(f)
    else:
        TypeError('buf is not a file name and it has no write '
                  ' method')

    if buf is None:
        return '<style type=text/css> {} </style>\n{}'.format(css_style_string,
                                                              formatter.buf.getvalue())


def format_rotated_headers(df, min_rotation_level=0, css_style_string='',
                           rotated_cell_height=120,
                           buf=None, columns=None, col_space=None, header=True,
                           index=True, na_rep='NaN', formatters=None,
                           float_format=None, sparsify=None, index_names=True,
                           justify=None,
                           bold_rows=True, classes=None,
                           max_rows=None, max_cols=None, show_dimensions=False,
                           notebook=False):
    """
    Returns a string containing the html-formatted DataFrame df with rotated
    (90 degree) headers and styled with the css style provided in
    css_style_string. If you want to style the whole table, you have to specify
    the css class to use in classes.

    For more information on the parameters see the pandas documentation for
    DataFrameFormatter (currently empty for this topic...).

    :param df: The DataFrame to be formatted
    :param min_rotation_level: If using a multiindexed DataFrame, only header
        fields with a higher level than this will be displayed rotated.
    :param css_style_string: CSS-Style to be embedded in the html document.
    """
    css_style_string += '''
th.rotate {
  white-space: nowrap;
  -webkit-transform-origin: 65px 60px;
  -moz-transform-origin: 65px 60px;
  -o-transform-origin: 65px 60px;
  -ms-transform-origin: 65px 60px;
  transform-origin: 65px 60px;
  -webkit-transform: rotate(270deg);
  -moz-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  -o-transform: rotate(270deg);
  transform: rotate(270deg);
}
span.intact {
  display: inline-block;
  width: 30px;
  height: %dpx;
}
''' % rotated_cell_height
    return format_html(RotatedTHeadHTMLFormatter, df, css_style_string, buf,
                       columns, col_space, header, index, na_rep, formatters,
                       float_format, sparsify, index_names, justify, bold_rows,
                       classes, max_rows, max_cols, show_dimensions, notebook,
                       min_rotation_level=min_rotation_level)
