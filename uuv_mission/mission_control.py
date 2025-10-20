# PD controller implemented as a small stateful class with reset() and step().
# This module is independent and can be imported by the simulator (ClosedLoop).
class PDController:
    def __init__(self, kp: float = 0.15, kd: float = 0.6, output_limits=None):
        # store proportional gain
        self.kp = kp
        # store derivative gain
        self.kd = kd
        # previous error, used to compute discrete derivative; initialize to 0.0
        self.prev_error = 0.0
        # optional tuple (min, max) to clip output; None means no clipping
        self.output_limits = output_limits

    def reset(self):
        # reset internal stored state so controller starts fresh between simulations
        self.prev_error = 0.0

    def step(self, reference: float, measurement: float, dt: float = 1.0) -> float:
        # compute the current error (r[t] - y[t])
        error = reference - measurement
        # compute derivative using backward difference, dividing by dt to account for sampling interval
        derivative = (error - self.prev_error) / dt if dt != 0 else 0.0
        # PD law: u = Kp * e + Kd * derivative
        u = self.kp * error + self.kd * derivative
        # update stored previous error for next step
        self.prev_error = error
        # if output limits provided, clip the action to those bounds
        if self.output_limits is not None:
            lo, hi = self.output_limits
            if lo is not None:
                u = max(lo, u)
            if hi is not None:
                u = min(hi, u)
        # return computed control action
        return u