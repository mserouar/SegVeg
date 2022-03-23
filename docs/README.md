# Sphinx documentation generation

## Building the documentation

Make sure the ``[dev]`` extra dependencies have been installed then generate the API doc if needed:

```shell
cd docs
make apidoc
```

You can update the API doc with the same command when adding or removing python files but you will
probably need to delete the ``docs/source/api`` directory first to avoid leftover.

Then generate the HTML documentation:

```shell
make html
```

If you need to force a new complete build of the HTML documentation, use the ``clean`` target first:

```shell
make clean
make html
```

The final HTML documentation can be found in the ``docs/build/html`` directory.

## Sphinx references

* [Basic RST syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
* [RST directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html)
