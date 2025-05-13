# env_config.py

ENV_CONFIG = {
    'grid_size': 10,
    'quadrant_size': 5,
    'max_steps': 200,
    'remap_interval': 100,
    'quadrant_profiles': ['low', 'low', 'low', 'high']
}

RISK_PROFILES = {
    'high': (0.25, 0.25, 0.5),  # (danger_pct, low_pct, safe_pct)
    'low': (0.05, 0.10, 0.85)
}
