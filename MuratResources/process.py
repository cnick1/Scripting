# pylint: disable=no-member
import os
from utils import utils
from utils import visualization
from utils import skeleton
import numpy as np


def IMAC_final(experiment, fname=None, save=False):
    ret = {"io": experiment["io"], "files": list()}
    for file in experiment["files"]:
        if file["experiment"]["batch"] == 1 and file["experiment"]["run"] == 2:
            print(file["experiment"]["trial"])
            if file["experiment"]["trial"] is not None:
                ret["files"].append(file)

    if save:
        utils.save_json(ret, fname)

    return ret


def visualize_2d(experiments, fname=None, save=False):
    for file in experiments["files"]:
        try:
            if file["jobs"]["triangulation"] is not True:
                continue
        except KeyError:
            continue

        file_id = file["id"]
        corresponding_id = file["corresponding"]
        corresponding = utils.find_file(corresponding_id, experiments)

        try:
            visualized_file = file["jobs"]["visualized"]
        except:
            visualized_file = False

        try:
            visualized_corresponding = corresponding["jobs"]["visualized"]
        except:
            visualized_corresponding = False

        if visualized_file is not True and visualized_corresponding is not True:  # noqa: E501
            print(file_id, corresponding_id)
            file["jobs"].update({"visualized": True})
            corresponding["jobs"].update({"visualized": True})
            triangulated_person_fname = "results/triangulation/{fid}-{cid}.json".format(fid=file["id"], cid=corresponding_id)  # noqa: E501

            data = utils.read_json(triangulated_person_fname)
            total_frames = data["total_frames"]
            person = np.asarray(data["people"]).astype(float)
            cg = np.nanmean(person, axis=1)
            print(cg[0:100:, 2])
            pass


    if save:
        utils.save_json(experiments, fname)

    return experiments

def calculate_extrinsics(experiments, fname=None, save=False):
    K1, d1, K2, d2, _ = utils.read_optics("optics.json")
    for file in experiments["files"]:
        ts = str(utils.get_timestamp_ms())
        extrinsics = {"left": None, "right": None}

        if file["IMAC38"]:
            corresponding_id = file["corresponding"]
            corresponding = utils.find_file(corresponding_id, experiments)

            anchor_frame = file["video"]["anchor_frame"]
            total_frames = file["video"]["total_frames"]

            frame_number = utils.random(0, total_frames-anchor_frame)
            if file["cam"] == "left":
                fname_l = os.path.join("data", str(file["jobs"]["timestamp"]),
                                    str(file["id"]),
                                    str(frame_number).zfill(6)+".png")
                fname_r = os.path.join("data", str(corresponding["jobs"]["timestamp"]),
                                    str(corresponding["id"]),
                                    str(frame_number).zfill(6)+".png")
            else:
                fname_r = os.path.join("data", str(file["jobs"]["timestamp"]),
                                    str(file["id"]),
                                    str(frame_number).zfill(6)+".png")
                fname_l = os.path.join("data", str(corresponding["jobs"]["timestamp"]),
                                    str(corresponding["id"]),
                                    str(frame_number).zfill(6)+".png")

            try:
                extrinsics_file = file["jobs"]["extrinsics"]
            except:
                extrinsics_file = False

            try:
                extrinsics_corresponding = corresponding["jobs"]["extrinsics"]  # noqa: E501
            except:
                extrinsics_corresponding = False

            if extrinsics_file is not True and extrinsics_corresponding is not True:  # noqa: E501
                try:
                    I_l = utils.read_img(fname_l)
                    I_r = utils.read_img(fname_r)

                    [T_l, T_r, T_b], c, _ = utils.calculate_extrinsics(I_l, I_r, K1, d1, K2, d2)  # noqa: E501

                    # TODO: you might want to inverse transformation matrices
                    extrinsics = {"left": T_l.tolist(),
                                  "right": T_r.tolist(),
                                  "board": T_b.tolist()}

                    file["jobs"].update({"extrinsics": True,
                                         "extrinsics_mat": extrinsics})
                    corresponding["jobs"].update({"extrinsics": True,
                                                  "extrinsics_mat": extrinsics})
                except Exception as e:
                    print(fname_l, fname_r, e)
                    file["jobs"].update({"extrinsics": False})
                    corresponding["jobs"].update({"extrinsics": False})

    if save:
        utils.save_json(experiments, fname)

    return experiments


def filter_singleperson(experiments, fname=None, save=False):
    retval = experiments.copy()
    retval["files"] = list()

    for file in experiments["files"]:
        if file["experiment"]["batch"] == 1:
            retval["files"].append(file)

    if save:
        utils.save_json(retval, fname)

    return retval


def add_video_timestamps(experiments):
    timestamps = utils.folders_in_folder("data")

    for timestamp in timestamps:
        idx = utils.folders_in_folder(timestamp)
        timestamp = int(os.path.basename(timestamp))

        for id in idx:
            id = int(os.path.basename(id))
            file = utils.find_file(id, experiments)
            file["jobs"].update({"timestamp": timestamp})

    return experiments


def create_experiment(folder1="left", folder2="right", fname=None, save=False):
    experiment = utils.create_experiment_metadata()
    files = utils.files_in_folder(folder1)
    print("Videos found in left:", len(files))

    for file in files:
        file_metadata = utils.create_file_metadata(file)
        experiment["files"].append(file_metadata)

    files = utils.files_in_folder(folder2)
    print("Videos found in right:", len(files))

    for file in files:
        file_metadata = utils.create_file_metadata(file)
        experiment["files"].append(file_metadata)

    if save:
        utils.save_json(experiment, fname)
    return experiment


def mark_experiments_for_IMAC(experiment, fname=None, save=False):
    sum = 0
    sum_frames = 0
    for file in experiment["files"]:
        batch = file["experiment"]["batch"]
        if batch == 1 or batch == 2:
            file["IMAC38"] = True
            sum += 1
            sum_frames += file["video"]["total_frames"]

    black_list = [842, 843, 844, 7, 8, 9, 853, 17, 861, 25, 862, 26, 869, 33, 873, 37]  # noqa: E501

    for file in experiment["files"]:
        for b in black_list:
            if b == file["id"]:
                file["IMAC38"] = False
                sum -= 1
                sum_frames -= file["video"]["total_frames"]

    print("Total number of videos to be processed:", sum)
    print("Total number of frames to be processed:", sum_frames)
    print("Total number of minutes of to be processed:", sum_frames/7200)

    if save:
        utils.save_json(experiment, fname)
    return experiment


def extract_frames(experiment, maxframes=None, step=1, fname=None, save=False):  # noqa: E501
    sum = 0

    for file in experiment["files"]:
        corresponding = utils.find_file(file["corresponding"], experiment)
        timestamp = utils.get_timestamp()
        try:
            frames_extracted = file["jobs"]["frame_extraction"]
        except:
            frames_extracted = False

        if file["IMAC38"] and not frames_extracted:
            print("Extracting: ", file["id"])
            utils.extract_frames(file, timestamp,
                                 maxframes=maxframes, step=step)

            file.update({"jobs": {"frame_extraction": True}})
            sum += 1
            print("Completed: ", sum)

        try:
            frames_extracted = corresponding["jobs"]["frame_extraction"]
        except:
            frames_extracted = False

        if corresponding["IMAC38"] and not frames_extracted:
            print("Extracting: ", corresponding["id"], "corresponding.")
            utils.extract_frames(corresponding, timestamp,
                                 maxframes=maxframes, step=step)
            corresponding.update({"jobs": {"frame_extraction": True}})
            sum += 1
            print("Completed: ", sum)

    if save:
        utils.save_json(experiment, fname)
    return experiment


def triangulate_single(experiments, fname=None, save=False):
    optics = utils.read_json("optics.json")

    for file in experiments["files"]:
        if file["IMAC38"]:
            corresponding_id = file["corresponding"]
            corresponding = utils.find_file(corresponding_id, experiments)
            output_fname = "results/triangulation/{fid}-{cid}.json".format(fid=file["id"], cid=corresponding_id)

            try:
                triangulated_file = file["jobs"]["triangulation"]
            except Exception as e:
                triangulated_file = False

            try:
                extrinsics_file = file["jobs"]["extrinsics"]
            except Exception as e:
                extrinsics_file = False


            try:
                triangulated_corresponding = corresponding["jobs"]["triangulation"]  # noqa: E501
            except:
                triangulated_corresponding = False

            try:
                extrinsics_corresponding = corresponding["jobs"]["extrinsics"]
            except:
                extrinsics_corresponding = False

            if triangulated_file is not True and triangulated_corresponding is not True and extrinsics_file is True and extrinsics_corresponding is True:  # noqa: E501
                extrinsics = file["jobs"]["extrinsics_mat"]
                person_data = utils.triangulate_single(file, corresponding,
                                                       optics, extrinsics)
                utils.save_json(person_data, output_fname)

                file["jobs"].update({"triangulation": True,
                                     "triangulation_metadata": output_fname})
                corresponding["jobs"].update({"triangulation": True,
                                              "triangulation_metadata": output_fname})
    if save:
        utils.save_json(experiments, fname)

    return experiments


if __name__ == '__main__':
    pass
