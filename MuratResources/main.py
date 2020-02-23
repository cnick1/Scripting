import process
from utils import utils


def main():
    # experiment = process.create_experiment(folder1="raw_data_pilot_3/woscreen",# noqa: E501
    #                                        folder2="raw_data_pilot_3/wscreen",
    #                                        fname="metadata/pilot_3.json",
    #                                        save=True)
    experiment = utils.read_json("metadata/pilot_3.json")  # noqa: E501
    experiment = process.extract_frames(experiment, maxframes=1200,
                                        fname="metadata/pilot_3_frames.json",
                                        save=True)
    experiment = process.add_video_timestamps(experiment)
    utils.save_json(experiment, "metadata/pilot_3_frames.json")
    # experiment = process.filter_singleperson(experiment, "metadata/IMAC_singleperson.json", save=True)  # noqa: E501
    # experiment = utils.read_json("metadata/IMAC_singleperson.json")  # noqa: E501
    # experiment = utils.read_json("metadata/IMAC_singleperson.json")  # noqa: E501
    # experiment = process.IMAC_final(experiment, "metadata/IMAC_KEV.json", save=True)  # noqa: E501
    # experiment = utils.read_json("metadata/IMAC_KEV.json")  # noqa: E501
    # experiment = process.calculate_extrinsics(experiment, "metadata/IMAC_singleperson_extrinsics_camera_origin_with_board.json", True)  # noqa: E501
    # experiment = utils.read_json("metadata/IMAC_singleperson_extrinsics_camera_origin_with_board.json")  # noqa: E501
    # experiment = process.triangulate_single(experiment, "metadata/IMAC_singleperson_triangulated__.json", True)  # noqa: E501


if __name__ == '__main__':
    main()
