from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.pipeline.column_mapping import ColumnMapping
import pandas as pd
from typing import Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load reference and current data
    reference_data = pd.read_parquet("reference_data.parquet")
    current_data = pd.read_parquet("current_data.parquet")
    return reference_data, current_data

def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    column_mapping = ColumnMapping()
    
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
    
    # Save drift report
    dashboard.save("drift_report.html")

def main():
    reference_data, current_data = load_data()
    detect_drift(reference_data, current_data)

if __name__ == "__main__":
    main()
