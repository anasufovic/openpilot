import math
import numpy as np

from cereal import log
from opendbc.car.honda.values import CAR as HONDA
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController
from openpilot.common.realtime import DT_CTRL

# NRDR modified-EPS speed-banded feedforward for the Civic Bosch on the TGG-A120 4250 image. night-star
# applies this at runtime (selfdrive/controls/lib/latcontrol_pid.py NRDR_MODIFIED_EPS_KF_*, 84f1b7ec) on
# top of the scalar kf 3.6e-6 that CarParams carries; this schema has no kfBP/kfV, so it lives here.
# Duplicate-near-25 mph breakpoint preserves the hard hand-off of the kp/ki schedule.
_MPH = 0.44704
EPS_MOD_KF_BP = [0.0, 25.0 * _MPH - 1e-3, 25.0 * _MPH, 50.0 * _MPH]
EPS_MOD_KF_V = [2.4e-6, 1.8e-6, 3.6e-6, 6.0e-6]

# Turn-phase detector, verbatim from nrdr/openpilot nrdr-nightly efe4f758 openpilot/sunnypilot/nrdr/phase_detector.py
# (night-star has the same function). phase > 0: the desired angle is winding into a turn; phase < 0: unwinding
# back toward center. The direction is latched below 0.5 mph so a stopped car keeps its last phase.
EPS_MOD_PHASE_SWITCH_MIN_SPEED = 0.5 * _MPH
# Reversal decay (see update()): bleed a stored integral that opposes the error during unwind, but only after the
# unwind phase has persisted this long, so model dither at an intersection cannot strip a needed integral.
EPS_MOD_UNWIND_I_DECAY_TAU_S = 0.25
EPS_MOD_UNWIND_PERSIST_S = 0.2


def phase_with_latch(angle: float, angle_delta: float, v_ego: float, direction: float) -> tuple[float, float]:
  phase = angle * angle_delta
  if phase != 0.0 and (v_ego > EPS_MOD_PHASE_SWITCH_MIN_SPEED or direction == 0.0):
    direction = 1.0 if phase > 0.0 else -1.0
  return abs(phase) * direction, direction


class LatControlPID(LatControl):
  def __init__(self, CP, CP_SP, CI):
    super().__init__(CP, CP_SP, CI)
    self.pid = PIDController((CP.lateralTuning.pid.kpBP, CP.lateralTuning.pid.kpV),
                             (CP.lateralTuning.pid.kiBP, CP.lateralTuning.pid.kiV),
                             k_f=CP.lateralTuning.pid.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.get_steer_feedforward = CI.get_steer_feedforward_function()
    self.eps_mod = CP.carFingerprint == HONDA.HONDA_CIVIC_BOSCH and bool(CP_SP.flags & HondaFlagsSP.EPS_MODIFIED)
    if self.eps_mod:
      self.prev_angle_steers_des = 0.0
      self.phase_direction = 0.0
      self.unwind_frames = 0

  def reset(self):
    super().reset()
    if self.eps_mod:
      # controlsd calls this every frame lateral is inactive. Drop the PID state so a stale integrator is not
      # re-injected at the next engagement (the modified-EPS fade-up would otherwise mask it for 1 s and then
      # apply it in full).
      self.pid.reset()
      self.phase_direction = 0.0
      self.unwind_frames = 0

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
      if self.eps_mod:
        # keep the phase detector's history current while inactive (nrdr does the same), so the first active
        # frame does not see a false wind/unwind step from a stale desired angle
        self.prev_angle_steers_des = angle_steers_des_no_offset

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

      if self.eps_mod:
        # Unwind handling for the modified-EPS Civic Bosch. Drive 2 (2026-08-27, route 005d227007) showed the
        # integrator winding to -0.38 in a long turn and then cancelling P for 2-3 s after the model reversed
        # ("green line moves, car does nothing"). Two rules, both only while the PID is not otherwise frozen
        # (driver override, < 5 m/s):
        #  1. nrdr's unwind freeze (nrdr-nightly latcontrol_pid.py:312): while unwinding, don't let the integrator
        #     grow if it already pushes the same way as the error.
        #  2. Reversal decay (ours, not in the reference; James/nrdr get their return from output scaling and an
        #     unwind feedforward boost instead): once the unwind phase has persisted 0.2 s and the stored integral
        #     opposes the current error, bleed it toward zero with a 0.25 s time constant. This shrinks |i| only,
        #     but removing an opposing term raises the net P+I+F output (still clipped to +-1): the intent is to
        #     hand authority back to P and FF on a reversal instead of letting a stale integral cancel them.
        #     Offline replay of the drive-2 bookmark windows: command 0.5-1.5 s after the reversal goes from
        #     +0.10/+0.19 to +0.27/+0.41.
        angle_delta = angle_steers_des_no_offset - self.prev_angle_steers_des
        phase, self.phase_direction = phase_with_latch(angle_steers_des_no_offset, angle_delta, CS.vEgo, self.phase_direction)
        # persistence only accumulates while the PID is live: a frozen stretch (override, < 5 m/s) must not
        # pre-arm the decay for the moment the freeze lifts
        self.unwind_frames = self.unwind_frames + 1 if (phase < 0.0 and not freeze_integrator) else 0
        if phase < 0.0 and not freeze_integrator:
          if self.pid.i * error > 0.0:
            freeze_integrator = True
          elif self.pid.i * error < 0.0 and self.unwind_frames * DT_CTRL >= EPS_MOD_UNWIND_PERSIST_S:
            self.pid.i *= math.exp(-DT_CTRL / EPS_MOD_UNWIND_I_DECAY_TAU_S)
        self.prev_angle_steers_des = angle_steers_des_no_offset

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
