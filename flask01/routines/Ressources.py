#  File to easily define new poses and workouts
import routines.Workout as Workout
import routines.Pose as Pose
import routines.Transition as Transition

# define poses

START_POSE = Pose.Pose("NO_PATH", 0, 0, True)
TREE_POSE = Pose.Pose("Tree.jpg", 25, 1, True)
COBRA_POSE = Pose.Pose("CobraPose.png", 25, 1, True)
DOWNDOG_POSE = Pose.Pose("DownwardFacingDog.png", 25, 1, True)
FLATBACK_POSE = Pose.Pose("FlatBack.png", 25, 1, True)

# define workouts

# TEST_WORKOUT = Workout.Workout('TEST_WORKOUT', poses=[START_POSE, TREE_POSE, COBRA_POSE, DOWNDOG_POSE, FLATBACK_POSE])
TEST_WORKOUT = Workout.Workout('TEST_WORKOUT', poses=[START_POSE, TREE_POSE,
                                                      Pose.Pose("RaisedArmsPose.png", 20, 1, True)])

SUN_SALUTATION = Workout.Workout('SUN_SALUTATION', poses=[START_POSE,
                                                          Pose.Pose("MountainPose.png", 10, 1, True),
                                                          Pose.Pose("RaisedArmsPose.png", 10, 1, True),
                                                          Pose.Pose("StandingForwardBend.png", 10, 1, True),
                                                          Pose.Pose("FlatBack.png", 10, 1, True),
                                                          Pose.Pose("PlankPose.png", 10, 1, True),
                                                          Pose.Pose("KneesChestAndChinPose.png", 10, 1, True),
                                                          Pose.Pose("CobraPose.png", 10, 1, True),
                                                          Pose.Pose("DownwardFacingDog.png", 10, 1, True),
                                                          Pose.Pose("StandingForwardBend.png", 10, 1, True),
                                                          Pose.Pose("RaisedArmsPose.png", 10, 1, True),
                                                          Pose.Pose("MountainPose.png", 10, 1, True),
                                                          ])

DISPLAY_SAMPLE1 = Workout.Workout('DISPLAY_SAMPLE1', poses=[START_POSE,
                                                            Pose.Pose("MountainPose.png", 10, 1, True),
                                                            Pose.Pose("RaisedArmsPose.png", 10, 1, True),
                                                            Pose.Pose("StandingForwardBend.png", 10, 1, True),
                                                            Pose.Pose("FlatBack.png", 10, 1, True),
                                                            Pose.Pose("PlankPose.png", 10, 1, True),
                                                            Pose.Pose("KneesChestAndChinPose.png", 10, 1, True),
                                                            Pose.Pose("CobraPose.png", 10, 1, True),
                                                            Pose.Pose("DownwardFacingDog.png", 10, 1, True),
                                                            ])

DISPLAY_SAMPLE2 = Workout.Workout('DISPLAY_SAMPLE2', poses=[START_POSE,
                                                            Pose.Pose("FlatBack.png", 10, 1, True),
                                                            Pose.Pose("PlankPose.png", 10, 1, True),
                                                            Pose.Pose("KneesChestAndChinPose.png", 10, 1, True),
                                                            Pose.Pose("CobraPose.png", 10, 1, True),
                                                            Pose.Pose("StandingForwardBend.png", 10, 1, True),
                                                            Pose.Pose("RaisedArmsPose.png", 10, 1, True),
                                                            Pose.Pose("MountainPose.png", 10, 1, True),
                                                            ])

DEBUG_WORKOUT = Workout.Workout('DEBUG_WORKOUT', poses=[START_POSE, Pose.Pose("Tree.jpg", 20, 1, True), DOWNDOG_POSE])

WORKOUT_3D = Workout.Workout('WORKOUT_3D', poses=[START_POSE,
                                                  Pose.Pose("Tree.jpg", 20, 1, True),
                                                  Pose.Pose("MountainPose.png", 20, 1, True),
                                                  Pose.Pose("RaisedArmsPose.png", 20, 1, True)
                                                  ])

VERYWELL_WORKOUT = Workout.Workout('VERYWELL_WORKOUT',
                                   poses=[START_POSE,
                                          Pose.Pose("p1.png", 15, 1, True),
                                          Pose.Pose("p2.png", 15, 1, True),
                                          Pose.Pose("p3.png", 15, 1, True),
                                          ],
                                   transitions=[Transition.Transition("tr_before_p1.mp4", 13),
                                                Transition.Transition("tr_before_p2.mp4", 33),
                                                Transition.Transition("tr_before_p3.mp4", 45)
                                                ])

# -------------------------------------------------------------------------------------------------------------------- #

workouts = [TEST_WORKOUT, SUN_SALUTATION, DISPLAY_SAMPLE1, DISPLAY_SAMPLE2, WORKOUT_3D]

# -------------------------------------------------------------------------------------------------------------------- #
