"""Single-attempt judges for RoboTwin tasks.

RoboTwin's own protocol keeps an episode running until `check_success` holds or the step limit
is reached, so a policy that misses can retry and still count as a success. A judge ends the
episode as a failure once the first attempt is over without success, so every success in the
replay is a first-try success.
"""


class HammerStrikeJudge:
    """beat_block_hammer: one strike, judged where the hammer head comes down.

    Heights are those of the hammer's striking point (functional point 0) above the block's centre
    (functional point 1); the block top is at 0.025 m. Expert replays raise the head to 0.06-0.07 m,
    cross 0.035 m one step before success and succeed at 0.025 m; misses keep falling past the block
    top beside it.
    """

    # The head is clearly above the block top: the strike is under way.
    ARM_HEIGHT = 0.045
    # Below the block top by 1 cm: the head came down beside the block (or the hammer fell).
    MISS_HEIGHT = 0.015
    # Near the block top: the head must reach success within MAX_SETTLE_STEPS env steps after this.
    STRIKE_HEIGHT = 0.035
    MAX_SETTLE_STEPS = 10

    def __init__(self):
        self.armed = False
        self.strike_step = None

    def attempt_failed(self, task):
        """Return True once the first strike has ended without success."""
        head = task.hammer.get_functional_point(0, "pose").p
        block = task.block.get_functional_point(1, "pose").p
        height = head[2] - block[2]
        if not self.armed:
            self.armed = height > self.ARM_HEIGHT
            return False
        if height < self.STRIKE_HEIGHT and self.strike_step is None:
            self.strike_step = task.take_action_cnt
        if height < self.MISS_HEIGHT:
            return True
        if self.strike_step is None:
            return False
        return task.take_action_cnt - self.strike_step >= self.MAX_SETTLE_STEPS


ATTEMPT_JUDGES = {"beat_block_hammer": HammerStrikeJudge}
