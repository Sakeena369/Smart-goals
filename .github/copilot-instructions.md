## Managing Dependencies
We use UV to manage python dependencies for this project. The `uv.lock` file contains the exact versions of all dependencies, ensuring reproducibility across different environments.

To install the required dependencies for this project, you can use the following command:

```bash
uv sync
```

To add new dependencies, use 

```bash
uv add <package-name>
```


