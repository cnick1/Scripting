# pylint: disable=no-member
import os
from collections import OrderedDict, namedtuple
import json
import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import random


class utils(object):
    """Provide utility tools as all static methods."""

    def __init__(self):
        super(utils, self).__init__()

    @staticmethod
    def draw_matchsticks(people, fname, frame_width=640, frame_height=480):
        f = plt.figure(figsize=plt.figaspect(1))
        ax = f.add_subplot(1, 1, 1)
        f.set_figwidth(8)
        f.set_figheight(6)
        ax.set_xlim([0, frame_width])
        ax.set_ylim([0, frame_height])
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")
        ax.invert_yaxis()

        if len(people.list) > 0:
            ax.set_title("Person Localization Frame: " +
                         str(people.list[0].frame))

        for person in people.list:
            for bone_id, bone in enumerate(person.bone_list):
                color = person.bone_colors[bone_id].ravel()

                if not np.any(person.joint_locs[bone] == 0):
                    ax.plot([person.joint_locs[bone[0], 0],
                             person.joint_locs[bone[1], 0]],
                            [person.joint_locs[bone[0], 1],
                             person.joint_locs[bone[1], 1]],
                            color=color)

        f.savefig(fname)

    @staticmethod
    def read_img(fname):
        img = cv2.imread(fname)
        if img is None:
            raise IOError("The img does not exist!")
        return img

    @staticmethod
    def save_img(fname, img):
        retval = cv2.imwrite(fname, img)
        if not retval:
            folder = os.path.dirname(fname)
            utils.create_folder(fname)
            retval = cv2.imwrite(fname, img)
            if not retval:
                raise IOError("I cannot write the image")

    @staticmethod
    def read_optics(fname):
        optics = utils.read_json(fname)
        K1 = np.asarray(optics["left"]["K"])
        K2 = np.asarray(optics["right"]["K"])
        d1 = np.asarray(optics["left"]["d"])
        d2 = np.asarray(optics["right"]["d"])
        return K1, d1, K2, d2, optics

    @staticmethod
    def calculate_extrinsics(I_l, I_r, K1, d1, K2, d2):
        try:
            text_l, _, canvas_l, R_l, t_l = utils.calculate_camera_pose(I_l, K1, d1)  # noqa: E501
        except Exception as e:
            raise e

        try:
            text_r, _, canvas_r, R_r, t_r = utils.calculate_camera_pose(I_r, K2, d2)  # noqa: E501
        except Exception as e:
            raise e

        canvas = np.concatenate([canvas_l, canvas_r], axis=1)
        R_l_inv = R_l.T
        t_l_inv = -R_l_inv.dot(t_l)
        rvec_l_inv, _ = cv2.Rodrigues(R_l_inv)
        rvec_r, _ = cv2.Rodrigues(R_r)
        rvec_l_r, t_vec_l_r, _, _, _, _, _, _, _, _ = cv2.composeRT(rvec_l_inv, t_l_inv, rvec_r, t_r)  # noqa: E501
        R_r_, _ = cv2.Rodrigues(rvec_l_r)
        T_l = utils.transformation_mat_from_R_t(np.eye(3), np.zeros([3, 1]))
        T_r = utils.transformation_mat_from_R_t(R_r_, t_vec_l_r)
        T_b = utils.transformation_mat_from_R_t(R_l, t_l)
        return [T_l, T_r, T_b], canvas, [text_l, text_r]

    @staticmethod
    def transformation_mat_from_R_t(R, t):
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return T

    @staticmethod
    def undistort_points(points, K, dist):
        """Undistort points to eliminate the lens distortion."""
        return cv2.undistortPoints(points, K, dist)

    @staticmethod
    def _calculate_camera_pose(frame, K, d, corners, pattern_shape=(6, 4), grid_size=30):  # noqa: E501
        """Calculate camera pose with a frame containing checkerboard in it."""
        img = frame.copy()
        axis = np.float32([[grid_size, 0, 0], [0, grid_size, 0],
                           [0, 0, -grid_size]]).reshape(-1, 3)*2

        objp = np.zeros((np.prod(pattern_shape), 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_shape[0],
                               0:pattern_shape[1]].T.reshape(-1, 2) * grid_size

        _, rvecs, tvecs = cv2.solvePnP(objp, corners, K, d)
        R, _ = cv2.Rodrigues(rvecs)
        # project 3D points onto image plane
        imgpts, _ = cv2.projectPoints(axis,
                                      rvecs, tvecs,
                                      K, d)

        canvas = utils.draw_axis(img, corners, imgpts)
        return R, tvecs, canvas

    @staticmethod
    def calculate_camera_pose(frame,  K, d, pattern_shape=(6, 4), grid_size=75):  # noqa: E501
        """Calculate camera pose with a frame containing checkerboard in it."""
        frame = cv2.undistort(frame, K, d)

        try:
            corners, canvas = utils.detect_chessboard(frame, pattern_shape)  # noqa: E501
        except Exception as e:
            raise e

        # canvas = cv2.undistort(canvas, K, d)
        R, t, canvas = utils._calculate_camera_pose(frame, K, d, corners,
                                                    pattern_shape,
                                                    grid_size)
        text = " ".join(np.round(t, 2).ravel().astype(str))
        return text, corners, canvas, R, t

    @staticmethod
    def detect_chessboard(frame, pattern_shape=(7, 6)):
        """Detect chessboard with a given shape in the frame."""
        corners = None
        canvas = None
        img = frame.copy()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 3000, 0.001)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # gray = clahe.apply(gray)
        ret, corners = cv2.findChessboardCorners(gray, pattern_shape)

        if ret:
            corners = cv2.cornerSubPix(gray, corners,
                                       (11, 11), (-1, -1), criteria)
            canvas = cv2.drawChessboardCorners(img, pattern_shape,
                                               corners, ret)
        else:
            raise ValueError("Checkerboard is not found")

        return corners, canvas

    @staticmethod
    def draw_axis(img, corners, imgpts):
        """Draw 3D axis on a given image."""
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    @staticmethod
    def triangulate_single(file, corresponding, optics, extrinsics):
        timestamp = file["jobs"]["timestamp"]
        K1 = np.asarray(optics["left"]["K"])
        K2 = np.asarray(optics["right"]["K"])
        d1 = np.asarray(optics["left"]["d"])
        d2 = np.asarray(optics["right"]["d"])

        # print(K1, d1)
        # print(K2, d2)

        RT1 = np.asarray(extrinsics["left"])
        RT2 = np.asarray(extrinsics["right"])
        # delete the part needed for homogenous stuff
        RT1 = np.delete(RT1, 3, axis=0)
        RT2 = np.delete(RT2, 3, axis=0)

        # print(RT1)
        # print(RT2)

        P1 = K1.dot(RT1)
        P2 = K2.dot(RT2)

        if file["cam"] == "left":
            id_l = file["id"]
            id_r = corresponding["id"]
        else:
            id_l = corresponding["id"]
            id_r = file["id"]

        path_l = os.path.join("data", str(timestamp), str(id_l), "results/")
        path_r = os.path.join("data", str(timestamp), str(id_r), "results/")

        results_l = utils.files_in_folder(path_l, ".json")
        results_r = utils.files_in_folder(path_r, ".json")
        results_l.sort()
        results_r.sort()

        n_frames = np.min([len(results_l), len(results_r)])
        results_l = results_l[:n_frames]
        results_r = results_r[:n_frames]

        results_2d = zip(results_l, results_r)
        results_3d = np.full((n_frames, 25, 3), None)

        for frame_id, result in enumerate(results_2d):
            path_l = result[0]
            path_r = result[1]
            people_l = utils.read_json(path_l)['people']
            people_r = utils.read_json(path_r)['people']

            # print("L #:", len(people_l),
            #       "R #:", len(people_r))

            if len(people_l) == 0 or len(people_r) == 0:
                continue

            if len(people_l) == 1:
                pose_l = np.asarray(people_l[0]['pose_keypoints_2d']).reshape(-1, 3)
            else:
                pose_l = list()
                for person in people_l:
                    xy = np.asarray(person['pose_keypoints_2d']).reshape(-1, 3)
                    pose_l.append(xy)

                # print("Left CGs:")
                cgs = np.mean(pose_l, axis=1)
                # print(cgs)

                most_right_idx = np.argmax(cgs[:, 0])
                pose_l = np.asarray(people_l[most_right_idx]['pose_keypoints_2d']).reshape(-1, 3)

            if len(people_r) == 1:
                pose_r = np.asarray(people_r[0]['pose_keypoints_2d']).reshape(-1, 3)
            else:
                pose_r = list()
                for person in people_r:
                    xy = np.asarray(person['pose_keypoints_2d']).reshape(-1, 3)
                    pose_r.append(xy)

                # print("Right CGs:")
                cgs = np.mean(pose_r, axis=1)
                # print(cgs)

                most_right_idx = np.argmax(cgs[:, 0])
                pose_r = np.asarray(people_r[most_right_idx]['pose_keypoints_2d']).reshape(-1, 3)

            joint_exists_l = pose_l[:, 2] > 0.1
            joint_exists_r = pose_r[:, 2] > 0.1

            both_exists = np.logical_and(joint_exists_l, joint_exists_r)

            if np.sum(both_exists) > 0:
                pose_l_subset = cv2.undistortPoints(pose_l[both_exists, 0:2].reshape(1, -1, 2), K1, d1, P=K1)  # noqa: E501
                pose_r_subset = cv2.undistortPoints(pose_r[both_exists, 0:2].reshape(1, -1, 2), K2, d2, P=K2)  # noqa: E501

                person_3d = utils.triangulate_points(P1, P2, pose_l_subset, pose_r_subset)[0:3, :]  # noqa: E501
                results_3d[frame_id, both_exists, :] = person_3d.T

        retval = {"total_frames": int(n_frames), "people": results_3d.tolist(),
                  "id_left": id_l, "id_right": id_r}
        return retval

    @staticmethod
    def triangulate_points(P1, P2, x1, x2):
        X = cv2.triangulatePoints(P1, P2, x1, x2)
        return X/X[3]

    @staticmethod
    def find_file(file_id, experiment):
        for file in experiment["files"]:
            if file_id == file["id"]:
                return file
        raise Exception(file_id, " does not exist in the experiment metadata!")

    @staticmethod
    def get_timestamp():
        return str(int(time.time()))

    @staticmethod
    def get_timestamp_ms():
        return str(int(time.time()*1000))

    @staticmethod
    def create_experiment_metadata(file_prefix="GOPR", file_extension=".MP4",
                                   number_of_digits=4, left_folder="left",
                                   right_folder="right", files=list()):
        """Create a metadata for the whole experiment."""
        metadata = OrderedDict(
            {
                "io": {
                    "file_prefix": file_prefix,
                    "file_extension": file_extension,
                    "number_of_digits": number_of_digits,
                    "left_folder": left_folder,
                    "right_folder": right_folder
                },
                "files": files
            }
        )
        return metadata

    @staticmethod
    def create_file_metadata_empty(id, video_file, camera, corresponding_id,
                                   run, batch, trial,
                                   anchor_frame, total_frames, IMAC38=False,
                                   people=list(), label=None):
        """Create metadata for each video file."""
        metadata = OrderedDict({"id": id,
                                "file": video_file,
                                "cam": camera,
                                "label": label,
                                "IMAC38": IMAC38,
                                "corresponding": corresponding_id,
                                "video": {
                                    "anchor_frame": anchor_frame,
                                    "total_frames": total_frames,
                                },
                                "people": people,
                                "experiment": {
                                    "run": run,
                                    "batch": batch,
                                    "trial": trial}
                                })
        return metadata

    @staticmethod
    def create_file_metadata(fname):
        id = utils.video_id_from_filename(os.path.basename(fname))
        run = 0
        batch = 0
        trial = 0
        camera = os.path.dirname(fname)
        anchor_frame, total_frames, label = utils.get_video_properties(fname)
        run, batch, trial, corresponding_id = label
        return utils.create_file_metadata_empty(id, fname, camera,
                                                corresponding_id, run, batch,
                                                trial, anchor_frame,
                                                total_frames)

    @staticmethod
    def get_video_properties(fname):
        id = utils.video_id_from_filename(os.path.basename(fname))
        video = cv2.VideoCapture(fname)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        corresponding_id, anchor_frame, run, batch, trial = utils.hardcoded_values(id)  # noqa: E501
        return anchor_frame, total_frames, [run, batch, trial, corresponding_id]  # noqa: E501

    @staticmethod
    def check_fie_exist(arg):
        pass

    @staticmethod
    def create_folder(fname):
        """Create an empty folder."""
        os.makedirs(fname)

    @staticmethod
    def check_folder_exists(folder):
        """Check if folder given in the argument exists or not."""
        return os.path.isdir(folder)

    @staticmethod
    def files_in_folder(folder, ext=".MP4"):
        file_list = list()
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(ext):
                    file_ = os.path.join(root, file)
                    file_list.append(file_)
        return file_list

    @staticmethod
    def folders_in_folder(folder):
        folders = list()
        for item in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, item)):
                folders.append(os.path.join(folder, item))
        return folders

    @staticmethod
    def video_id_from_filename(fname):
        if len(fname) != 12:
            raise IOError("Wrong format in filename!")
        else:
            return int(fname[4:-4])

    @staticmethod
    def filenamefrom_video_id(id, metadata):
        pass

    @staticmethod
    def read_json(fname):
        """Read data from a json file."""
        try:
            with open(fname) as file_handler:
                data = json.load(file_handler, object_pairs_hook=OrderedDict)
        except Exception as e:
            print(e)
            raise
        return data

    @staticmethod
    def save_json(data, file, pretty=True):
        """Write data to a json file."""
        with open(file, 'w') as outfile:
            if pretty:
                json.dump(data, outfile, indent=4, sort_keys=True)
            else:
                json.dump(data, outfile)

    @staticmethod
    def random(min, max):
        return random.randint(min, max)

    @staticmethod
    def extract_frames(file, timestamp, maxframes=None, step=1):
        video = cv2.VideoCapture(file["file"])
        video.set(cv2.CAP_PROP_POS_FRAMES, float(file["video"]["anchor_frame"]))  # noqa: E501
        remaining = file["video"]["total_frames"] - file["video"]["anchor_frame"]  # noqa: E501
        folder = os.path.join("data", timestamp, str(file["id"]))

        if not utils.check_folder_exists(folder):
            utils.create_folder(folder)

        for i in range(0, remaining, step):
            retval, image = video.read()
            if not retval:
                image = np.zeros((720, 1280, 3), dtype=np.uint8)

            cv2.imwrite("{folder}/{fn}.png".format(folder=folder, fn=str(i).zfill(6)), image)  # noqa: E501

            if maxframes is not None:
                if i>=maxframes:
                    break

    @staticmethod
    def hardcoded_values(id):
        corresponding_id, anchor_frame, run, batch, trial = None, None, None, None, None  # noqa: E501
        idx = [None, None, None, 846, 847, 848, 849, 850, 851, 852, 853, 854,
               855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867,
               868, 869, 870, 871, 872, 873, 874,
               None, None, None, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38]

        anchor_frames = [None, None, None, 3324, 2578, 578, 3458, 1418, 2607,
                         2755, 4207, 1846, 942, 2324, 2348, None, 1243, 1664,
                         None, 4293, 4195, 1897, 2803, 6355, 1419, 1435, 1924,
                         6003, 1413, 2022, None, 5100,
                         None, None, None, 2432, 2330, 229, 1800, 1161, 2320,
                         2416, 3926, 1411, 653, 2066, 2591, None, 1923, 1584,
                         None, 4004, 3817, 1574, 2516, 6603, 1178, 1025, 1889,
                         6356, 1073, 1754, None, 5798]

        left = [842, 843, 844, 846, 847, 848, 849, 850, 851, 852, 853, 854,
                855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866,
                867, 868, 869, 870, 871, 872, 873, 874]
        right = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                 38]

        batches = [1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 6, 6, 5, 5, 1,
                   1, 1, 1, 1, 4, 2, 3, 3, 1, 1, 1, 1]

        runs = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 1, 1, 1, 3, 3, 3,
                3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2]

        trials = [1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 4, 1, 1, 1, 2, 1, 1,
                  1, 2, 3, 4, 1, 1, 1, 1, 1, 2, None, 3]

        for _id, elem in enumerate(idx):
            if elem == id:
                anchor_frame = anchor_frames[_id]

        for _id, elem in enumerate(left):
            if elem == id:
                corresponding_id = right[_id]
                batch = batches[_id]
                run = runs[_id]
                trial = trials[_id]

        for _id, elem in enumerate(right):
            if elem == id:
                corresponding_id = left[_id]
                batch = batches[_id]
                run = runs[_id]
                trial = trials[_id]

        return corresponding_id, anchor_frame, run, batch, trial


class people(object):
    def __init__(self, json, frame=None):
        super(people, self).__init__()
        self.json = json
        self.json = np.asarray(self.json).reshape(-1, 25, 3)
        self.list = list()
        for person_id, person in enumerate(self.json):
            self.list.append(skeleton(person,
                                      person_id=person_id,
                                      frame=frame))


class skeleton(object):
    """This class implements stuff about the skeleton data structure."""

    def __init__(self, joint_data, person_id=None, frame=None):
        """Initialize the class."""
        super(skeleton, self).__init__()
        self.n_joint = 25
        self.frame = frame
        self.person_id = person_id
        self.joint_locs = np.array(joint_data)
        self.joint_locs = self.joint_locs.reshape(-1, 3)
        self.bone_list = np.array([[0, 16],
                                   [0, 15],
                                   [1,  0],
                                   [1,  8],
                                   [1,  5],
                                   [1,  2],
                                   [2, 17],
                                   [2,  3],
                                   [3,  4],
                                   [5,  6],
                                   [5, 18],
                                   [6,  7],
                                   [8,  9],
                                   [8, 12],
                                   [9, 10],
                                   [10, 11],
                                   [11, 22],
                                   [11, 24],
                                   [12, 13],
                                   [13, 14],
                                   [14, 19],
                                   [14, 21],
                                   [15, 17],
                                   [16, 18],
                                   [19, 20],
                                   [22, 23]])
        self.bone_colors = np.random.rand(len(self.bone_list), 3, 1)

        self.names = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                      "LShoulder", "LElbow", "LWrist", "MidHip",
                      "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                      "REye", "LEye", "REar", "LEar", "LBigToe",
                      "LSmallToe", "LHeel", "RBigToe", "RSmallToe",
                      "RHeel", "Background"]

        self.nose = joint(joint_id=0, name=self.names[0],
                          loc2d=self.joint_locs[0, 0:2],
                          conf=self.joint_locs[0, 2])
        self.neck = joint(joint_id=1, name=self.names[1],
                          loc2d=self.joint_locs[1, 0:2],
                          conf=self.joint_locs[1, 2])
        self.rshoulder = joint(joint_id=2, name=self.names[2],
                               loc2d=self.joint_locs[2, 0:2],
                               conf=self.joint_locs[2, 2])
        self.relbow = joint(joint_id=3, name=self.names[3],
                            loc2d=self.joint_locs[3, 0:2],
                            conf=self.joint_locs[3, 2])
        self.rwrist = joint(joint_id=4, name=self.names[4],
                            loc2d=self.joint_locs[4, 0:2],
                            conf=self.joint_locs[4, 2])
        self.lshoulder = joint(joint_id=5, name=self.names[5],
                               loc2d=self.joint_locs[5, 0:2],
                               conf=self.joint_locs[5, 2])
        self.lelbow = joint(joint_id=6, name=self.names[6],
                            loc2d=self.joint_locs[6, 0:2],
                            conf=self.joint_locs[6, 2])
        self.lwrist = joint(joint_id=7, name=self.names[7],
                            loc2d=self.joint_locs[7, 0:2],
                            conf=self.joint_locs[7, 2])
        self.midhip = joint(joint_id=8, name=self.names[8],
                            loc2d=self.joint_locs[8, 0:2],
                            conf=self.joint_locs[8, 2])
        self.rhip = joint(joint_id=9, name=self.names[9],
                          loc2d=self.joint_locs[9, 0:2],
                          conf=self.joint_locs[9, 2])
        self.rknee = joint(joint_id=10, name=self.names[10],
                           loc2d=self.joint_locs[10, 0:2],
                           conf=self.joint_locs[10, 2])
        self.rankle = joint(joint_id=11, name=self.names[11],
                            loc2d=self.joint_locs[11, 0:2],
                            conf=self.joint_locs[11, 2])
        self.lhip = joint(joint_id=12, name=self.names[12],
                          loc2d=self.joint_locs[12, 0:2],
                          conf=self.joint_locs[12, 2])
        self.lknee = joint(joint_id=13, name=self.names[13],
                           loc2d=self.joint_locs[13, 0:2],
                           conf=self.joint_locs[13, 2])
        self.lankle = joint(joint_id=14, name=self.names[14],
                            loc2d=self.joint_locs[14, 0:2],
                            conf=self.joint_locs[14, 2])
        self.reye = joint(joint_id=15, name=self.names[15],
                          loc2d=self.joint_locs[15, 0:2],
                          conf=self.joint_locs[15, 2])
        self.leye = joint(joint_id=16, name=self.names[16],
                          loc2d=self.joint_locs[16, 0:2],
                          conf=self.joint_locs[16, 2])
        self.rear = joint(joint_id=17, name=self.names[17],
                          loc2d=self.joint_locs[17, 0:2],
                          conf=self.joint_locs[17, 2])
        self.lear = joint(joint_id=18, name=self.names[18],
                          loc2d=self.joint_locs[18, 0:2],
                          conf=self.joint_locs[18, 2])
        self.lbigtoe = joint(joint_id=19, name=self.names[19],
                             loc2d=self.joint_locs[19, 0:2],
                             conf=self.joint_locs[19, 2])
        self.lsmalltoe = joint(joint_id=20, name=self.names[20],
                               loc2d=self.joint_locs[20, 0:2],
                               conf=self.joint_locs[20, 2])
        self.lheel = joint(joint_id=21, name=self.names[21],
                           loc2d=self.joint_locs[21, 0:2],
                           conf=self.joint_locs[21, 2])
        self.rbigtoe = joint(joint_id=22, name=self.names[22],
                             loc2d=self.joint_locs[22, 0:2],
                             conf=self.joint_locs[22, 2])
        self.rsmalltoe = joint(joint_id=23, name=self.names[23],
                               loc2d=self.joint_locs[23, 0:2],
                               conf=self.joint_locs[23, 2])
        self.rheel = joint(joint_id=24, name=self.names[24],
                           loc2d=self.joint_locs[24, 0:2],
                           conf=self.joint_locs[24, 2])


class joint(object):
    """This class implements stuff about the joint data structure."""

    def __init__(self, joint_id, name, loc2d=None, conf=None, loc3d=None):
        """Initialize the class."""
        super(joint, self).__init__()
        self.id = joint_id
        self.name = name
        self.loc2d = loc2d
        self.conf = conf
        self.loc3d = loc3d


class visualization(object):
    """docstring for visualization."""

    def __init__(self):
        """Creates a figure in which detected skeletons are drawn."""
        super(visualization, self).__init__()

    @staticmethod
    def draw_matchsticks(people, fname, frame_width=1280, frame_height=720):
        f = plt.figure(figsize=plt.figaspect(1))
        ax = f.add_subplot(1, 1, 1)
        f.set_figwidth(8)
        f.set_figheight(6)
        ax.set_xlim([0, frame_width])
        ax.set_ylim([0, frame_height])
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")
        ax.invert_yaxis()

        if len(people.list) > 0:
            ax.set_title("Person Localization Frame: " +
                         str(people.list[0].frame))

        for person in people.list:
            for bone_id, bone in enumerate(person.bone_list):
                color = person.bone_colors[bone_id].ravel()

                if not np.any(person.joint_locs[bone] == 0):
                    ax.plot([person.joint_locs[bone[0], 0],
                             person.joint_locs[bone[1], 0]],
                            [person.joint_locs[bone[0], 1],
                             person.joint_locs[bone[1], 1]],
                            color=color)

        f.savefig(fname)
        plt.close()

if __name__ == '__main__':
    pass
