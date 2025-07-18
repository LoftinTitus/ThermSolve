import pandas as pd
import json
from typing import List, Dict, Any, Optional

class IO:
    """
    Utility class for loading and saving substance/property data.
    """
    @staticmethod
    def load_substances_from_csv(path: str) -> pd.DataFrame:
        """Load substances from a CSV file."""
        return pd.read_csv(path)

    @staticmethod
    def load_substances_from_json(path: str) -> List[Dict[str, Any]]:
        """Load substances from a JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_substances_to_json(substances: List[Dict[str, Any]], path: str) -> None:
        """Save substances to a JSON file."""
        with open(path, 'w') as f:
            json.dump(substances, f, indent=2)

    @staticmethod
    def save_substances_to_csv(substances: pd.DataFrame, path: str) -> None:
        """Save substances to a CSV file."""
        substances.to_csv(path, index=False)

    @staticmethod
    def load_citation_metadata(path: str) -> Optional[Dict[str, Any]]:
        """Load citation/source metadata from a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def export_to_latex(substances: pd.DataFrame, path: str) -> None:
        """Export substances to a LaTeX table."""
        latex_str = substances.to_latex(index=False)
        with open(path, 'w') as f:
            f.write(latex_str)

    @staticmethod
    def export_to_table(substances: pd.DataFrame) -> str:
        """Return a human-readable table as a string."""
        return substances.to_string(index=False)