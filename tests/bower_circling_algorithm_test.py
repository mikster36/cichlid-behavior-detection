import unittest

import numpy as np

from behavior_detection.bower_circling import Fish, Vel
from behavior_detection.bower_circling import track_bower_circling as track_bower_circling


class BowerCirclingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.video = ""
        self.proximity = 12
        self.head_tail_proximity = 8
        self.threshold = 90
        self.track_age = 1
        self.bower_circling_length = 1
        self.extract_clips = False

    def test_multiple_tracks_same_fish(self):
        """
        Scenario: fish1 and fish2 are bc for 2 frames, fish2 leaves, fish1 bc with fish3 for 3 frames some time later,
        fish3 leaves, fish1 bc with fish 2 for 1 frame some time later.
        Expected outcome: 3 tracks -> {fish1 & fish2 start = frame001, end = frame002, length = 2}
                                   -> {fish1 & fish3 start = frame004, end = frame006, length = 3}
                                   -> {fish1 & fish2 start = frame008, end = frame008, length = 1}
        """
        self.frames = {'frame001': {'fish1': Fish(id='fish1',
                                                  position=[np.array([10, 10]), np.array([9, 10]),
                                                            np.array([8, 10])],
                                                  vel=[Vel(direction=np.array([0.7071, -0.7071]), magnitude=1),
                                                       Vel(direction=np.array([1, 0]), magnitude=1),
                                                       Vel(direction=np.array([0.7071, 0.7071]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([8, 5]), np.array([9, 5]),
                                                            np.array([10, 5])],
                                                  vel=[Vel(direction=np.array([-0.7071, 0.7071]), magnitude=1),
                                                       Vel(direction=np.array([-1, 0]), magnitude=1),
                                                       Vel(direction=np.array([-0.7071, -0.7071]), magnitude=1)],
                                                  bc=False)},
                       'frame002': {'fish1': Fish(id='fish1',
                                                  position=[np.array([11, 9]), np.array([10, 10]),
                                                            np.array([9, 11])],
                                                  vel=[Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([7, 6]), np.array([8, 5]), np.array([9, 4])],
                                                  vel=[Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1)],
                                                  bc=False)},
                       'frame003': {'fish1': Fish(id='fish1',
                                                  position=[np.array([11, 9]), np.array([10, 10]),
                                                            np.array([9, 11])],
                                                  vel=[Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([100, 100]), np.array([100, 100]), np.array([100, 100])],
                                                  vel=[Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1)],
                                                  bc=False)},
                       'frame004': {'fish1': Fish(id='fish1',
                                                  position=[np.array([10, 10]), np.array([9, 10]),
                                                            np.array([8, 10])],
                                                  vel=[Vel(direction=np.array([1, -1]), magnitude=1),
                                                       Vel(direction=np.array([1, 0]), magnitude=1),
                                                       Vel(direction=np.array([0.7071, 0.7071]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([100, 100]), np.array([100, 100]),
                                                            np.array([100, 100])],
                                                  vel=[Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1)],
                                                  bc=False),
                                    'fish3': Fish(id='fish3',
                                                  position=[np.array([8, 5]), np.array([9, 5]),
                                                            np.array([10, 5])],
                                                  vel=[Vel(direction=np.array([-0.7071, 0.7071]), magnitude=1),
                                                       Vel(direction=np.array([-1, 0]), magnitude=1),
                                                       Vel(direction=np.array([-0.7071, -0.7071]), magnitude=1)],
                                                  bc=False)},
                       'frame005': {'fish1': Fish(id='fish1',
                                                  position=[np.array([11, 9]), np.array([10, 10]),
                                                            np.array([9, 11])],
                                                  vel=[Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([100, 100]), np.array([100, 100]),
                                                            np.array([100, 100])],
                                                  vel=[Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1)],
                                                  bc=False),
                                    'fish3': Fish(id='fish3',
                                                  position=[np.array([7, 6]), np.array([8, 5]), np.array([9, 4])],
                                                  vel=[Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1)],
                                                  bc=False)},
                       'frame006': {'fish1': Fish(id='fish1',
                                                  position=[np.array([11, 8]), np.array([10, 9]),
                                                            np.array([9, 10])],
                                                  vel=[Vel(direction=np.array([-0.7071, -0.7071]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, 0]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([100, 100]), np.array([100, 100]),
                                                            np.array([100, 100])],
                                                  vel=[Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1)],
                                                  bc=False),
                                    'fish3': Fish(id='fish3',
                                                  position=[np.array([7, 7]), np.array([8, 6]), np.array([9, 5])],
                                                  vel=[Vel(direction=np.array([0.7071, 0.7071]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 0]), magnitude=1)],
                                                  bc=False)},
                       'frame007': {'fish1': Fish(id='fish1',
                                                  position=[np.array([11, 8]), np.array([10, 9]),
                                                            np.array([9, 10])],
                                                  vel=[Vel(direction=np.array([-0.7071, -0.7071]), magnitude=1),
                                                       Vel(direction=np.array([0, -1]), magnitude=1),
                                                       Vel(direction=np.array([0, 0]), magnitude=1)],
                                                  bc=False),
                                    'fish3': Fish(id='fish3',
                                                  position=[np.array([100, 100]), np.array([100, 100]), np.array([100, 100])],
                                                  vel=[Vel(direction=np.array([0.7071, 0.7071]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 0]), magnitude=1)],
                                                  bc=False)},
                       'frame008': {'fish1': Fish(id='fish1',
                                                  position=[np.array([10, 10]), np.array([9, 10]),
                                                            np.array([8, 10])],
                                                  vel=[Vel(direction=np.array([0.7071, -0.7071]), magnitude=1),
                                                       Vel(direction=np.array([1, 0]), magnitude=1),
                                                       Vel(direction=np.array([0.7071, 0.7071]), magnitude=1)],
                                                  bc=False),
                                    'fish2': Fish(id='fish2',
                                                  position=[np.array([8, 5]), np.array([9, 5]),
                                                            np.array([10, 5])],
                                                  vel=[Vel(direction=np.array([-0.7071, 0.7071]), magnitude=1),
                                                       Vel(direction=np.array([-1, 0]), magnitude=1),
                                                       Vel(direction=np.array([-0.7071, -0.7071]), magnitude=1)],
                                                  bc=False),
                                    'fish3': Fish(id='fish3',
                                                  position=[np.array([100, 100]), np.array([100, 100]),
                                                            np.array([100, 100])],
                                                  vel=[Vel(direction=np.array([0.7071, 0.7071]), magnitude=1),
                                                       Vel(direction=np.array([0, 1]), magnitude=1),
                                                       Vel(direction=np.array([0, 0]), magnitude=1)],
                                                  bc=False)},
                       }
        bc_incidents = track_bower_circling(video=self.video,
                                            frames=self.frames,
                                            proximity=self.proximity,
                                            head_tail_proximity=self.head_tail_proximity,
                                            threshold=self.threshold,
                                            track_age=self.track_age,
                                            bower_circling_length=self.bower_circling_length,
                                            extract_clips=self.extract_clips,
                                            debug=True)

        expected_bc_incidents = [{'a': 'fish1', 'b': 'fish2', 'start': 'frame001', 'end': 'frame002'},
                                 {'a': 'fish1', 'b': 'fish3', 'start': 'frame004', 'end': 'frame006'},
                                 {'a': 'fish1', 'b': 'fish2', 'start': 'frame008', 'end': 'frame008'}]
        for i in range(len(bc_incidents)):
            incident = bc_incidents[i]
            assert incident.a.id == expected_bc_incidents[i]['a']
            assert incident.b.id == expected_bc_incidents[i]['b']
            assert incident.start == expected_bc_incidents[i]['start']
            assert incident.end == expected_bc_incidents[i]['end']


if __name__ == '__main__':
    unittest.main()
