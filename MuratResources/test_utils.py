import unittest
from utils import utils

class test_utils(unittest.TestCase):
    def setUp(self):
        pass

    def test_videoid(self):
        u = utils()
        self.assertEqual(u.video_id_from_filename("GOPR0833.MP4"), 833)


if __name__ == '__main__':
    unittest.main()
