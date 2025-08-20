Absolutely 🚀 — creating and managing pipelines is the core workflow in Kedro. Here’s a **cheat sheet of the most important commands** you’ll use when working with pipelines:

---

## 🔧 Project Setup

```bash
kedro new            # Create a new Kedro project
kedro install        # Install dependencies from pyproject.toml
```

---

## 📦 Pipeline Commands

```bash
kedro pipeline create <pipeline_name>  
```

Creates a new pipeline folder with `pipeline.py`, `__init__.py`, and a `nodes.py` file.

Example:

```bash
kedro pipeline create data_processing
```

---

## 🏃 Running Pipelines

```bash
kedro run                            # Run the default pipeline (all registered pipelines)
kedro run --pipeline=<pipeline_name> # Run a specific pipeline
kedro run --node=<node_name>         # Run a specific node
```

Examples:

```bash
kedro run --pipeline=data_processing
kedro run --node=clean_data_node
```

---

## 📝 Node & Catalog

Nodes = Python functions, Pipelines = sequence of nodes.

* Add a node in `nodes.py`
* Wire it into the pipeline in `pipeline.py`
* Define inputs/outputs in `conf/base/catalog.yml`

---

## 🧪 Testing

```bash
kedro test      # Run tests with pytest
```

---

## 📊 Visualization (Optional, requires plugin)

```bash
pip install kedro-viz
kedro viz run
```

This launches a UI to explore your pipeline graph.

---

## 📚 Jupyter / IPython Integration

```bash
kedro jupyter notebook   # Start Jupyter Notebook with Kedro context
kedro jupyter lab        # Start Jupyter Lab
kedro ipython            # Start IPython shell with Kedro context
```

---

### 🛠️ Typical Workflow

1. `kedro pipeline create data_processing`
2. Edit `nodes.py` → write your function(s).
3. Edit `pipeline.py` → wire nodes together.
4. Update `conf/base/catalog.yml` → define datasets.
5. Run:

   ```bash
   kedro run --pipeline=data_processing
   ```

---

👉 Do you want me to also draft a **minimal example pipeline** (load CSV → clean data → save output) that you can copy-paste as your starting point?
