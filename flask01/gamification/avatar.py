from enum import Enum  # for enum34, or the stdlib version

# from aenum import Enum  # for the aenum version

# TODO: find good thresholds
# TODO: show score ticks

Difficulty = Enum('difficulty', 'easy medium hard')


class YogAvatar:
    def __init__(self, difficulty):
        self.healthbar = 100
        self.score = 0
        self.dead = False
        self.combo = 1
        self.max_combo = 1
        self.saved = 0

        self.base = 5
        self.max_drain = self.base
        self.min_drain = -self.base
        self.threshold = None
        self.thresh_300 = None
        self.penalty = None
        self.few_points = None
        self.many_points = None
        self.difficulty = None
        self.set_difficulty(difficulty)
        self.verbose = False

    def set_difficulty(self, difficulty):
        if difficulty is Difficulty['easy']:
            self.threshold = 0.4
            self.thresh_300 = 0.3
            self.few_points = 50
            self.many_points = 150
            self.difficulty = str(difficulty)[11:]

        elif difficulty is Difficulty['medium']:
            self.threshold = 0.35
            self.thresh_300 = 0.25
            self.few_points = 100
            self.many_points = 300
            self.difficulty = str(difficulty)[11:]

        elif difficulty is Difficulty['hard']:
            self.threshold = 0.28
            self.thresh_300 = 0.18
            self.few_points = 150
            self.many_points = 450
            self.difficulty = str(difficulty)[11:]

        self.penalty = self.base / self.threshold

    def calculate_health_change(self, score):
        health = self.base - score * self.penalty
        if health > self.max_drain:
            health = self.max_drain
        if health < self.min_drain:
            health = self.min_drain
        return health

    def drain(self, score, enemy_combo=None):
        health = self.calculate_health_change(score)
        if health > 0:
            self.combo += .1
            if enemy_combo:
                health *= self.combo
            if self.combo > self.max_combo:
                self.max_combo = self.combo
        else:
            self.combo = 1
            if enemy_combo:
                health -= enemy_combo * self.base
            else:
                health *= 2
        self.healthbar += health
        if self.verbose:
            print(f"combo amplifier: x{self.combo}")
            print(f"new health: {self.healthbar} health change: {health}")
        self.update_score(score)
        if self.healthbar <= 0:
            self.healthbar = 0
            self.dead = True
            return False
        if self.healthbar >= 100:
            self.healthbar = 100
        return True

    def update_score(self, score):
        if self.dead:
            return
        if score <= self.thresh_300:
            self.score += int(self.many_points * self.combo)
            if self.verbose:
                print(f"+{self.many_points}")
        elif score <= self.threshold:
            self.score += int(self.few_points * self.combo)
            if self.verbose:
                print(f"+{self.few_points}")
        if self.verbose:
            print(f"new score: {self.score}")

    def reset(self):
        self.healthbar = 100
        self.score = 0
        self.dead = False
        self.combo = 1
        self.saved = 0
