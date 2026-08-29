"""Regression tests for the modified-EPS Civic Bosch additions to LatControlPID (banded kf, PID reset on
inactive, integrator freeze exception, unwind freeze + reversal decay). Only imports latcontrol_pid so it runs
without the prebuilt params/msgq binaries."""
import math

import numpy as np
import pytest

from cereal import log
from opendbc.car import structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.controls.lib.latcontrol_pid import (LatControlPID, phase_with_latch, EPS_MOD_KF_BP, EPS_MOD_KF_V,
                                                             EPS_MOD_UNWIND_PERSIST_S, EPS_MOD_DRIVER_HOLD_COUNTS)

FP = {0: {}, 1: {}, 2: {}}
MODDED_FW = b'39990-TGG,A120\x00\x00'
STOCK_FW = b'39990-TGG-A120\x00\x00'


class _VM:
  # linear plant for the tests: 1 deg of steer per 1e-3 1/m of curvature, speed independent
  def get_steer_from_curvature(self, curv, v_ego, roll):
    return math.radians(curv * 1000.0)


class _CS:
  def __init__(self, v_ego=8.0, angle=0.0, pressed=False, torque=None):
    self.vEgo = v_ego
    self.steeringAngleDeg = angle
    self.steeringRateDeg = 0.0
    self.steeringPressed = pressed
    self.steeringTorque = torque if torque is not None else (2500.0 if pressed else 0.0)


def _controller(fw):
  car_fw = [structs.CarParams.CarFw(ecu=structs.CarParams.Ecu.eps, fwVersion=fw, address=0x18DA30F1, subAddress=0)]
  CP = CarInterface.get_params(CAR.HONDA_CIVIC_BOSCH, FP, car_fw, False, False, False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.HONDA_CIVIC_BOSCH, FP, car_fw, False, False)
  return LatControlPID(CP, CP_SP, CarInterface(CP, CP_SP))


def _step(lc, active, cs, desired_deg, limited=False):
  params = log.LiveParametersData.new_message()
  out, _, _ = lc.update(active, cs, _VM(), params, limited, -desired_deg / 1000.0, None, False)
  return out


def _wind(lc, angle_des=60.0, lag=15.0, frames=300, v=8.0):
  for _ in range(frames):
    _step(lc, True, _CS(v, angle_des - lag), angle_des)


def test_gating():
  assert _controller(MODDED_FW).eps_mod
  assert not _controller(STOCK_FW).eps_mod


def test_stock_has_no_eps_state():
  lc = _controller(STOCK_FW)
  for attr in ("phase_direction", "unwind_frames", "prev_angle_steers_des"):
    assert not hasattr(lc, attr)


def test_banded_kf():
  lc = _controller(MODDED_FW)
  for mph, kf in ((0, 2.4e-6), (24.9, 1.8e-6), (25.0, 3.6e-6), (50, 6.0e-6), (70, 6.0e-6)):
    _step(lc, True, _CS(mph * 0.44704, 0.0), 1.0)
    assert lc.pid.k_f == pytest.approx(np.interp(mph * 0.44704, EPS_MOD_KF_BP, EPS_MOD_KF_V))
    assert lc.pid.k_f == pytest.approx(kf, rel=2e-3)


def test_reset_clears_pid_and_phase():
  lc = _controller(MODDED_FW)
  _wind(lc)
  assert abs(lc.pid.i) > 0.05
  lc.reset()
  assert lc.pid.i == 0.0 and lc.phase_direction == 0.0 and lc.unwind_frames == 0


def test_freeze_exception_only_for_modded():
  lc = _controller(MODDED_FW)
  for _ in range(100):
    _step(lc, True, _CS(15.0, 0.0), 3.0, limited=True)
  assert abs(lc.pid.i) > 0.01  # integrates despite steer_limited_by_safety
  lc2 = _controller(STOCK_FW)
  for _ in range(100):
    _step(lc2, True, _CS(15.0, 0.0), 0.5, limited=True)
  assert lc2.pid.i == 0.0  # stock: frozen


def test_inactive_frames_track_desired_angle():
  lc = _controller(MODDED_FW)
  for _ in range(50):
    lc.reset()
    _step(lc, False, _CS(8.0, 30.0), 30.0)
  assert lc.prev_angle_steers_des == pytest.approx(30.0)
  # first active frame with an unchanged desired angle: no false wind/unwind step
  _step(lc, True, _CS(8.0, 30.0), 30.0)
  assert lc.unwind_frames == 0 and lc.phase_direction == 0.0


def test_standstill_latch_not_seeded_by_stale_history():
  lc = _controller(MODDED_FW)
  _wind(lc, angle_des=60.0)
  for _ in range(50):
    lc.reset()  # controlsd calls LaC.reset() every frame lateral is inactive
    _step(lc, False, _CS(0.0, 10.0), 10.0)  # inactive at standstill, desired settled at 10
  _step(lc, True, _CS(0.0, 10.0), 10.0)
  assert lc.phase_direction == 0.0  # no phase from the 60 -> 10 history


def test_reversal_decay_bleeds_opposing_integral():
  lc = _controller(MODDED_FW)
  _wind(lc)
  i0 = lc.pid.i
  assert i0 > 0.2
  # model reverses: desired drops 60 -> 0 over 1 s while the car lags behind (error opposes stored I)
  for k in range(100):
    des = 60.0 * (1 - k / 100)
    _step(lc, True, _CS(8.0, min(des + 25.0, 60.0)), des)
  assert lc.pid.i < 0.05 * i0


def test_no_decay_before_persistence():
  lc = _controller(MODDED_FW)
  _wind(lc)
  i0 = lc.pid.i
  n = int(EPS_MOD_UNWIND_PERSIST_S / 0.01) - 2
  for k in range(n):
    des = 60.0 - 0.5 * k
    _step(lc, True, _CS(8.0, 60.0), des)  # unwinding with error opposing I, but not yet for 0.2 s
  # only normal integration of the (small, opposite-sign) error may have happened, not the 0.25 s decay
  # (decay over these frames would have removed ~50% of i0)
  assert i0 - lc.pid.i < 0.02 * i0


def test_dither_does_not_strip_integral():
  lc = _controller(MODDED_FW)
  _wind(lc)
  i0 = lc.pid.i
  # model dithers around the setpoint: desired steps DOWN four frames in a row (four consecutive unwind frames,
  # 60 -> 59.5 -> 59 -> 58.5 -> 58), then jumps back up (one wind frame), repeat. The unwind phase therefore
  # persists 4 frames (0.04 s) at a time, never the 0.2 s the decay needs; the error opposes I throughout.
  # The integrator may only move by plain ki*error integration (tracked below), never by the decay.
  expected = i0
  for k in range(400):
    des = 60.0 - 0.5 * (k % 5)
    _step(lc, True, _CS(8.0, 61.0), des)
    expected += (des - 61.0) * lc.pid.k_i * 0.01
  assert lc.pid.i == pytest.approx(expected, rel=0.02)
  # sanity: the same amplitude with 25 consecutive unwind frames (> 0.2 s) DOES decay well beyond integration
  lc2 = _controller(MODDED_FW)
  _wind(lc2)
  expected2 = lc2.pid.i
  for k in range(400):
    des = 60.0 - 0.5 * (k % 26)
    _step(lc2, True, _CS(8.0, 61.0), des)
    expected2 += (des - 61.0) * lc2.pid.k_i * 0.01
  assert lc2.pid.i < 0.5 * expected2


def test_decay_rate_matches_time_constant():
  lc = _controller(MODDED_FW)
  _wind(lc)
  # sustained unwind with the car well behind (error opposes I); after the 0.2 s persistence gate the integral
  # must follow exp(-t/0.25) within the extra normal integration of the opposing error
  for k in range(20):
    _step(lc, True, _CS(8.0, 60.0), 60.0 - 0.5 * k)
  i_gate = lc.pid.i
  for k in range(20, 45):
    _step(lc, True, _CS(8.0, 60.0), 60.0 - 0.5 * k)
  expected = i_gate * math.exp(-0.25 / 0.25)
  assert lc.pid.i < expected * 1.05
  assert lc.pid.i > expected * 0.6  # not dramatically faster than the documented time constant


def test_override_release_mid_unwind_does_not_decay_immediately():
  lc = _controller(MODDED_FW)
  _wind(lc)
  assert lc.pid.i > 0.2
  # unwinding slowly while the driver holds the wheel (override): I is cleared and persistence must not arm
  for k in range(100):
    _step(lc, True, _CS(8.0, 60.0, pressed=True), 60.0 - 0.03 * k)
    assert lc.pid.i == 0.0
  assert lc.unwind_frames == 0
  # driver lets go with the car ahead of the model (error opposes any I): I restarts from zero and only plain
  # integration of the small negative error may happen in the first 15 frames -- no decay path is armed
  for k in range(100, 115):
    _step(lc, True, _CS(8.0, 60.0), 60.0 - 0.03 * k)
  assert abs(lc.pid.i) < 0.01
  assert lc.unwind_frames <= 15


def test_phase_with_latch():
  assert phase_with_latch(10.0, 1.0, 5.0, 0.0) == (10.0, 1.0)
  assert phase_with_latch(10.0, -1.0, 5.0, 1.0) == (-10.0, -1.0)
  assert phase_with_latch(-10.0, -1.0, 5.0, 0.0) == (10.0, 1.0)
  assert phase_with_latch(10.0, -1.0, 0.1, 1.0) == (10.0, 1.0)  # latched below 0.5 mph


def test_no_integration_while_driver_holds_below_threshold():
  lc = _controller(MODDED_FW)
  # steady 10 deg error with the driver holding 1200 counts (below the 1800 override threshold): I must not wind
  for _ in range(300):
    _step(lc, True, _CS(15.0, 0.0, torque=1200.0), 10.0)
  assert lc.pid.i == 0.0
  # same error with a resting hand below the hold threshold: integrates normally
  lc2 = _controller(MODDED_FW)
  for _ in range(300):
    _step(lc2, True, _CS(15.0, 0.0, torque=EPS_MOD_DRIVER_HOLD_COUNTS - 100.0), 10.0)
  assert lc2.pid.i > 0.05


def test_override_clears_integrator_and_release_restarts_from_zero():
  lc = _controller(MODDED_FW)
  _wind(lc)
  assert lc.pid.i > 0.2
  _step(lc, True, _CS(8.0, 45.0, pressed=True), 60.0)
  assert lc.pid.i == 0.0
  for _ in range(20):
    _step(lc, True, _CS(8.0, 45.0, pressed=True), 60.0)
  assert lc.pid.i == 0.0
  _step(lc, True, _CS(8.0, 45.0), 60.0)  # released
  assert 0.0 < lc.pid.i < 0.01


def test_stock_integrates_through_driver_hold():
  lc = _controller(STOCK_FW)
  for _ in range(100):
    _step(lc, True, _CS(15.0, 0.0, torque=1200.0), 0.5)
  assert lc.pid.i > 0.0
