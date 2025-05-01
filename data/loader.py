import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

def load_dataset(name="artificial"):
    if name == "artificial":
        np.random.seed(42)
        x_np = np.linspace(0, 1, 100).astype(np.float32)
        y_np = 3.0 * x_np + 2.0 + 0.1 * np.random.randn(100).astype(np.float32)
    elif name == "house_prices":
        dataset = fetch_openml(name="house_prices", version=1, as_frame=True)
        df = dataset.frame
        x_np = df['GrLivArea'].values.astype(np.float32)
        y_np = df['SalePrice'].values.astype(np.float32)
        #----- cleaning  NaNs
        mask = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_np = x_np[mask]
        y_np = y_np[mask]
        #----- basic scaling
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min())
        y_np = (y_np - y_np.min()) / (y_np.max() - y_np.min())
    else:
        raise ValueError(f"Dataset '{name}' no soportado.")
    
    return x_np, y_np
