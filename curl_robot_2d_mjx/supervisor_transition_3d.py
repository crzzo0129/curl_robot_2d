"""Runtime policy routing for ROLL -> TRANSITION -> WALK."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from curl_robot_2d_mjx.config_transition_3d import (
    ReadyToWalkSample3D,
    Transition3DConfig,
    is_ready_to_walk_3d,
    ready_to_walk_reasons_3d,
)


class PolicyRoute3D(str, Enum):
    ROLL = "roll"
    TRANSITION = "transition"
    WALK = "walk"


@dataclass(frozen=True)
class SupervisorOutput3D:
    route: PolicyRoute3D
    ready_hold_s: float
    ready: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class RoutedPolicyAction3D:
    route: PolicyRoute3D
    action: object
    supervisor: SupervisorOutput3D


class RollToWalkSupervisor3D:
    """Irreversible, debounce-protected high-level policy switch.

    A stop request transfers authority from ROLL to TRANSITION.  WALK receives
    authority only after every READY condition remains true for ``ready_hold_s``.
    Reset is explicit, so noisy estimates cannot accidentally return to ROLL.
    """

    def __init__(self, config: Transition3DConfig | None = None):
        self.config = config or Transition3DConfig()
        self.reset()

    def reset(self) -> None:
        self.route = PolicyRoute3D.ROLL
        self._ready_hold_s = 0.0

    def update(
        self,
        *,
        stop_requested: bool,
        sample: ReadyToWalkSample3D,
        dt: float,
    ) -> SupervisorOutput3D:
        if dt <= 0.0:
            raise ValueError("supervisor dt must be positive")
        if self.route is PolicyRoute3D.ROLL and stop_requested:
            self.route = PolicyRoute3D.TRANSITION

        gate_failures = ready_to_walk_reasons_3d(sample, self.config)
        gate_ready = is_ready_to_walk_3d(sample, self.config)
        if self.route is PolicyRoute3D.TRANSITION:
            self._ready_hold_s = (
                self._ready_hold_s + dt if gate_ready else 0.0
            )
            if self._ready_hold_s + 1.0e-12 >= self.config.ready_hold_s:
                self.route = PolicyRoute3D.WALK
        elif self.route is PolicyRoute3D.ROLL:
            self._ready_hold_s = 0.0

        return SupervisorOutput3D(
            route=self.route,
            ready=self.route is PolicyRoute3D.WALK,
            ready_hold_s=self._ready_hold_s,
            gate_failures=gate_failures,
        )


class ThreePolicyController3D:
    """Small deployment adapter that invokes the currently authorized policy.

    Each callable owns its native observation contract: ROLL may consume its
    existing 8-DoF observation/action interface, while TRANSITION and WALK use
    their 12-DoF interfaces.  The downstream actuator adapter remains explicit.
    """

    def __init__(
        self,
        *,
        roll_policy: Callable[[object], object],
        transition_policy: Callable[[object], object],
        walk_policy: Callable[[object], object],
        config: Transition3DConfig | None = None,
    ):
        self.policies = {
            PolicyRoute3D.ROLL: roll_policy,
            PolicyRoute3D.TRANSITION: transition_policy,
            PolicyRoute3D.WALK: walk_policy,
        }
        self.supervisor = RollToWalkSupervisor3D(config)

    def reset(self) -> None:
        self.supervisor.reset()

    def control(
        self,
        *,
        stop_requested: bool,
        ready_sample: ReadyToWalkSample3D,
        observations: Mapping[PolicyRoute3D | str, object],
        dt: float,
    ) -> RoutedPolicyAction3D:
        status = self.supervisor.update(
            stop_requested=stop_requested,
            sample=ready_sample,
            dt=dt,
        )
        observation = observations.get(status.route)
        if observation is None:
            observation = observations.get(status.route.value)
        if observation is None:
            raise KeyError(f"missing observation for {status.route.value} policy")
        action = self.policies[status.route](observation)
        return RoutedPolicyAction3D(
            route=status.route,
            action=action,
            supervisor=status,
        )
