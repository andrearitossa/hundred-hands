# X500 limits

Source: `x500*.model.sdf`, vendored from [PX4-gazebo-models](https://github.com/PX4/PX4-gazebo-models) (BSD-3, see `LICENSE`).

| Quantity | Value | From |
| --- | --- | --- |
| Mass m | 2.064 kg | base 2.0 + 4 rotors x 0.016 |
| Thrust per motor | 8.55 N | `motorConstant` 8.54858e-6 x (`maxRotVelocity` 1000)^2 |
| Total thrust T | 34.19 N | 4 motors; T/W = 1.69 |
| Max level horizontal accel | 13.3 m/s² | sqrt(T² − (mg)²) / m |
| Max climb accel | 6.75 m/s² | (T − mg) / m |

Limits keep 60 % of that authority, leaving the rest for attitude control: `a_max = 8`, `az_max = 4`.

The SDF has no drag or velocity limit, so `v_max = 15` and `vz_max = 5` are **not derived**. They remain the
spec's assumed values.
