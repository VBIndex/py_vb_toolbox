## Documentation setup

The documentation for this project is built with Sphinx.

To be able to contribute to the docs, you need to:

1. Create and activate a virtual environment with a recent Python version
2. Clone the repository
4. Install the developer requirements with `pip`:

```
pip install -r docs/requirements.txt
```

After making changes to the docs, you can build the HTML files from the `docs` directory with:

```
make html
```

The resulting file at `docs/_build/html/index.html` can be opened in your local browser to view your changes. Once the build succeeds locally, create a pull request with your changes.