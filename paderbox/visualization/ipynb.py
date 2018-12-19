import re
import types
from ipywidgets import widgets
from IPython.display import display, display_html, HTML


# http://matthiaseisen.com/pp/patterns/p0063/
def callback_button(callback_fcn, description="Click me!"):
    btn = widgets.Button(description=description)

    def func(btn):
        callback_fcn()
    btn.on_click(func)
    display(btn)


def toggle_code_button():
    # This line will hide code by default when the notebook is exported as HTML
    display_html(
        '<script>'
        'jQuery(function() '
        '{if (jQuery("body.notebook_app").length == 0) '
        '{ jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});'
        '</script>',
        raw=True)
    # This line will add a button to toggle visibility of code blocks, for use with the HTML export version
    display_html(
        '''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''',
        raw=True)


import ipywidgets as widgets  # Loads the Widget framework.
from IPython.core.magics.namespace import NamespaceMagics  # Used to query namespace.

# For this example, hide these names, just to avoid polluting the namespace further
#get_ipython().user_ns_hidden['widgets'] = widgets
#get_ipython().user_ns_hidden['NamespaceMagics'] = NamespaceMagics


class VariableInspectorWindow(object):
    instance = None

    def __init__(self, ipython,
                 regex_ignore=r'(VariableInspectorWindow|inspector)',
                 ignore_types=(types.ModuleType, types.FunctionType)):
        """Public constructor."""
        if VariableInspectorWindow.instance is not None:
            raise Exception("""Only one instance of the Variable Inspector can exist at a
                time.  Call close() on the active instance before creating a new instance.
                If you have lost the handle to the active instance, you can re-obtain it
                via `VariableInspectorWindow.instance`.""")

        VariableInspectorWindow.instance = self
        self.closed = False
        self.namespace = NamespaceMagics()
        self.namespace.shell = ipython.kernel.shell

        self._box = widgets.Box()
        self._box._dom_classes = ['inspector']
        self._box.background_color = '#fff'
        self._box.border_color = '#ccc'
        self._box.border_width = 1
        self._box.border_radius = 5

        self._modal_body = widgets.VBox()
        self._modal_body.overflow_y = 'scroll'

        self._modal_body_label = widgets.HTML(value = 'Not hooked')
        self._modal_body.children = [self._modal_body_label]

        self._box.children = [
            self._modal_body,
        ]

        self._ipython = ipython
        self._ipython.events.register('post_run_cell', self._fill)

        self.regex_ignore = regex_ignore
        self.ignore_types = ignore_types



    def close(self):
        """Close and remove hooks."""
        if not self.closed:
            self._ipython.events.unregister('post_run_cell', self._fill)
            self._box.close()
            self.closed = True
            VariableInspectorWindow.instance = None

    def _fill(self):
        """Fill self with variable information."""
        values = self.namespace.who_ls()

        def get_type(var):
            try:
                return type(var).__name__ + ' ' + str(var.dtype)
            except AttributeError:
                return type(var).__name__

        r = re.compile(self.regex_ignore)
        self._modal_body_label.value = \
            '<table class="table table-bordered table-striped"><tr><th>Name</th><th>Type</th><th>Value</th></tr><tr><td>' + \
            '</td></tr><tr><td>'.join(
                ['{0}</td><td>{1}</td><td>{2}'.format(
                    v,
                    get_type(self._ipython.user_ns[v]),
                    str(self._ipython.user_ns[v]))
                    for v in values if not r.match(v) and not isinstance(self._ipython.user_ns[v], self.ignore_types)]) + \
            '</td></tr></table>'

    def _ipython_display_(self):
        """Called when display() or pyout is used to display the Variable
        Inspector."""
        self._box._ipython_display_()

    def undock(self):
        from IPython.display import HTML
        from IPython.display import Javascript
        return Javascript(
        '''
        $('div.inspector')
            .detach()
            .prependTo($('body'))
            .css({
                'z-index': 999,
                position: 'fixed',
                'box-shadow': '5px 5px 12px -3px black',
                opacity: 0.9
            })
            .draggable();
        ''')
