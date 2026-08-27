import math
import numpy as np

from cereal import log
from opendbc.car.honda.values import CAR as HONDA
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController

# NRDR modified-EPS speed-banded feedforward for the Civic Bosch on the TGG-A120 4250 image. night-star
# applies this at runtime (selfdrive/controls/lib/latcontrol_pid.py NRDR_MODIFIED_EPS_KF_*, 84f1b7ec) on
# top of the scalar kf 3.6e-6 that CarParams carries; this schema has no kfBP/kfV, so it lives here.
# Duplicate-near-25 mph breakpoint preserves the hard hand-off of the kp/ki schedule.
_MPH = 0.44704
EPS_MOD_KF_BP = [0.0, 25.0 * _MPH - 1e-3, 25.0 * _MPH, 50.0 * _MPH]
EPS_MOD_KF_V = [2.4e-6, 1.8e-6, 3.6e-6, 6.0e-6]


class LatControlPID(LatControl):
  def __init__(self, CP, CP_SP, CI):
    super().__init__(CP, CP_SP, CI)
    self.pid = PIDController((CP.lateralTuning.pid.kpBP, CP.lateralTuning.pid.kpV),
                             (CP.lateralTuning.pid.kiBP, CP.lateralTuning.pid.kiV),
                             k_f=CP.lateralTuning.pid.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.get_steer_feedforward = CI.get_steer_feedforward_function()
    self.eps_mod = CP.carFingerprint == HONDA.HONDA_CIVIC_BOSCH and bool(CP_SP.flags & HondaFlagsSP.EPS_MODIFIED)

  def reset(self):
    super().reset()
    if self.eps_mod:
      # controlsd calls this every frame lateral is inactive. Drop the PID state so a stale integrator is not
      # re-injected at the next engagement (the modified-EPS fade-up would otherwise mask it for 1 s and then
      # apply it in full).
      self.pid.reset()

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg
    error = angle_steers_des - CS.steeringAngleDeg

    pid_log.steeringAngleDesiredDeg = angle_steers_des
    pid_log.angleError = error
    if not active:
      output_torque = 0.0
      pid_log.active = False

    else:
      # offset does not contribute to resistive torque
      if self.eps_mod:
        self.pid.k_f = float(np.interp(CS.vEgo, EPS_MOD_KF_BP, EPS_MOD_KF_V))
      ff = self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)
      # On the modified-EPS Civic Bosch the carcontroller deliberately shapes the command (override fade, LPF),
      # so actuatorsOutput.torque differs from actuators.torque on most frames and steer_limited_by_safety is
      # set although nothing limited us (45% of active frames on the 2026-08-27 drive). Don't let that starve
      # the integrator; steeringPressed and the 5 m/s floor still freeze it. Bounded I growth is possible during
      # the 1 s fade-up windows above 5 m/s (<= ~0.2 at ki 0.02 and 10 deg error).
      freeze_integrator = (steer_limited_by_safety and not self.eps_mod) or CS.steeringPressed or CS.vEgo < 5

      output_torque = self.pid.update(error,
                                feedforward=ff,
                                speed=CS.vEgo,
                                freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(output_torque)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return output_torque, angle_steers_des, pid_log
