

class Workout:
    def __init__(self, name, poses, transitions=None):
        self.name = name
        self.poses = poses
        self.transitions = transitions
        self.is_next_pose = 1
        self.is_next_transition = 0

# iterators

    def has_next_pose(self):
        try:
            i = self.poses[self.is_next_pose]
            return True
        except IndexError:
            return False
        # return self.isNext < len(self.poses)-1

    def next_pose(self):
        if not self.has_next_pose():
            return None
        self.is_next_pose += 1
        return self.poses[self.is_next_pose - 1]

    def has_next_transition(self):
        if self.transitions is None:
            return False
        try:
            i = self.transitions[self.is_next_transition + 1]
            return True
        except IndexError:
            return False

    def next_transition(self):
        if not self.has_next_transition():
            return None
        self.is_next_transition += 1
        return self.transitions[self.is_next_transition]

# getters

    def get_pose(self):
        return self.poses[self.is_next_pose - 1]

    def get_pose_count(self):
        return len(self.poses)

    def get_transition(self):
        if self.transitions is None:
            return None
        return self.transitions[self.is_next_transition]

    def get_transition_count(self):
        if self.transitions is None:
            return 0
        return len(self.transitions)

    def get_duration(self):
        duration = 0
        for p in self.poses:
            duration += p.duration
        return duration

# util

    def reset(self):
        self.is_next_pose = 1
        self.is_next_transition = 0
        return self
