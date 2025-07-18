import os
import tempfile
import pandas as pd
import json
import pytest
from thermsolve.io import IO

def test_load_and_save_csv():
    df = pd.DataFrame({"name": ["water", "ethanol"], "cp": [4180, 2440]})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name
    try:
        IO.save_substances_to_csv(df, path)
        loaded = IO.load_substances_from_csv(path)
        assert loaded.equals(df)
    finally:
        os.remove(path)

def test_load_and_save_json():
    substances = [{"name": "water", "cp": 4180}, {"name": "ethanol", "cp": 2440}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        IO.save_substances_to_json(substances, path)
        loaded = IO.load_substances_from_json(path)
        assert loaded == substances
    finally:
        os.remove(path)

def test_load_citation_metadata():
    metadata = {"source": "NIST", "year": 2020}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        with open(path, "w") as f:
            json.dump(metadata, f)
        loaded = IO.load_citation_metadata(path)
        assert loaded == metadata
    finally:
        os.remove(path)

def test_export_to_latex_and_table():
    df = pd.DataFrame({"name": ["water", "ethanol"], "cp": [4180, 2440]})
    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tmp:
        path = tmp.name
    try:
        IO.export_to_latex(df, path)
        with open(path) as f:
            latex = f.read()
        assert "\\begin{tabular}" in latex
    finally:
        os.remove(path)
    table_str = IO.export_to_table(df)
    assert "water" in table_str and "ethanol" in table_str
