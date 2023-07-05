# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "Xorbits"
copyright = "2022-2023, Xorbits Inc."
author = "xorbitsai"

from xorbits import __version__

release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

# i18n
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional


# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "image_light": "logo.svg",
        "image_dark": "logo-white.svg",
    },
}

# tags is injected by sphinx,
# see https://stackoverflow.com/a/73497480
if "zh_cn" not in tags.tags.keys():
    # en
    html_theme_options["external_links"] = [
        {"name": "xorbits.io", "url": "https://xorbits.io"},
    ]
    html_theme_options["icon_links"] = [
        {
            "name": "GitHub",
            "url": "https://github.com/xorbitsai/xorbits",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Slack",
            "url": "https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg",
            "icon": "fa-brands fa-slack",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/xorbitsio",
            "icon": "fa-brands fa-twitter",
            "type": "fontawesome",
        },
    ]
else:
    # zh_cn
    html_theme_options["external_links"] = [
        {"name": "xorbits.cn", "url": "https://xorbits.cn"},
    ]
    html_theme_options["icon_links"] = [
        {
            "name": "GitHub",
            "url": "https://github.com/xorbitsai/xorbits",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Gitee",
            "url": "https://gitee.com/xorbitsai/xorbits",
            "icon": "https://gitee.com/static/images/logo-en.svg",
            "type": "url",
        },
        {
            "name": "Zhihu",
            "url": "https://zhihu.com/org/xorbits",
            "icon": "fa-brands fa-zhihu",
            "type": "fontawesome",
        },
    ]

html_favicon = "_static/favicon.svg"


# remove the docstring of the flags attribute (inherited from numpy ndarray)
# because these give doc build errors (see GH issue 5331)
def remove_flags_docstring(app, what, name, obj, options, lines):
    if what == "attribute" and name.endswith(".flags"):
        del lines[:]


def process_class_docstrings(app, what, name, obj, options, lines):
    """For those classes for which we use ::

    :template: autosummary/class_without_autosummary.rst

    the documented attributes/methods have to be listed in the class
    docstring. However, if one of those lists is empty, we use 'None',
    which then generates warnings in sphinx / ugly html output.
    This "autodoc-process-docstring" event connector removes that part
    from the processed docstring.

    """
    if what == "class":
        joined = "\n".join(lines)

        templates = [
            """.. rubric:: Attributes

.. autosummary::
   :toctree:

   None
""",
            """.. rubric:: Methods

.. autosummary::
   :toctree:

   None
""",
        ]

        for template in templates:
            if template in joined:
                joined = joined.replace(template, "")
        lines[:] = joined.split("\n")


import sphinx  # isort:skip
from sphinx.ext.autodoc import (  # isort:skip
    AttributeDocumenter,
    Documenter,
    MethodDocumenter,
)
from sphinx.ext.autosummary import Autosummary  # isort:skip


class AccessorDocumenter(MethodDocumenter):
    """
    Specialized Documenter subclass for accessors.
    """

    objtype = "accessor"
    directivetype = "method"

    # lower than MethodDocumenter so this is not chosen for normal methods
    priority = 0.6

    def format_signature(self):
        # this method gives an error/warning for the accessors, therefore
        # overriding it (accessor has no arguments)
        return ""


class AccessorLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on accessor level (methods,
    attributes).
    """

    # This is the simple straightforward version
    # modname is None, base the last elements (eg 'hour')
    # and path the part before (eg 'Series.dt')
    # def resolve_name(self, modname, parents, path, base):
    #     modname = 'pandas'
    #     mod_cls = path.rstrip('.')
    #     mod_cls = mod_cls.split('.')
    #
    #     return modname, mod_cls + [base]
    def resolve_name(self, modname, parents, path, base):
        if modname is None:
            if path:
                mod_cls = path.rstrip(".")
            else:
                mod_cls = None
                # if documenting a class-level object without path,
                # there must be a current class, either from a parent
                # auto directive ...
                mod_cls = self.env.temp_data.get("autodoc:class")
                # ... or from a class directive
                if mod_cls is None:
                    mod_cls = self.env.temp_data.get("py:class")
                # ... if still None, there's no way to know
                if mod_cls is None:
                    return None, []
            # HACK: this is added in comparison to ClassLevelDocumenter
            # mod_cls still exists of class.accessor, so an extra
            # rpartition is needed
            modname, _, accessor = mod_cls.rpartition(".")
            modname, _, cls = modname.rpartition(".")
            parents = [cls, accessor]
            # if the module name is still missing, get it like above
            if not modname:
                modname = self.env.temp_data.get("autodoc:module")
            if not modname:
                if sphinx.__version__ > "1.3":
                    modname = self.env.ref_context.get("py:module")
                else:
                    modname = self.env.temp_data.get("py:module")
            # ... else, it stays None, which means invalid
        return modname, parents + [base]


class AccessorAttributeDocumenter(AccessorLevelDocumenter, AttributeDocumenter):
    objtype = "accessorattribute"
    directivetype = "attribute"

    # lower than AttributeDocumenter so this is not chosen for normal
    # attributes
    priority = 0.6


class AccessorMethodDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    objtype = "accessormethod"
    directivetype = "method"

    # lower than MethodDocumenter so this is not chosen for normal methods
    priority = 0.6


class AccessorCallableDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    """
    This documenter lets us removes .__call__ from the method signature for
    callable accessors like Series.plot
    """

    objtype = "accessorcallable"
    directivetype = "method"

    # lower than MethodDocumenter; otherwise the doc build prints warnings
    priority = 0.5

    def format_name(self):
        return MethodDocumenter.format_name(self).rstrip(".__call__")


def setup(app):
    app.connect("autodoc-process-docstring", remove_flags_docstring)
    app.connect("autodoc-process-docstring", process_class_docstrings)
    app.add_autodocumenter(AccessorDocumenter)
    app.add_autodocumenter(AccessorAttributeDocumenter)
    app.add_autodocumenter(AccessorMethodDocumenter)
    app.add_autodocumenter(AccessorCallableDocumenter)
