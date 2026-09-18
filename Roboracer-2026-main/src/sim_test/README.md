# sim_test

Simulation-only package.

## Launch

- Mapping:
  - `ros2 launch sim_test sim_mapping.launch.py`
- Pure localization:
  - `ros2 launch sim_test sim_localization.launch.py pbstream_filename:=/absolute/path/map.pbstream`

## Notes

- This package is separated from `localization_layer` (real stack).
- Sensor noise and rates are managed in `config/sim_sensor_params.yaml`.
- If `centerline_csv_path` is set, the fake vehicle follows that centerline path first.
