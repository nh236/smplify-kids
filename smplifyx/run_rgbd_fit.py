# Copyright©2023 University Children’s Hospital Zurich – Eleonore Foundation
# By using this code you agree to the terms specified in the LICENSE file

from rgbd_fit.fit_rgbd_sequence import run_fitting

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True, type=str, help='Top folder contating the data.')
    parser.add_argument('--output_folder', required=True, type=str, help='Folder where the results are stored')
    parser.add_argument('--gender', required=False, default='male', choices=['female', 'male', 'neutral'],
                        type=str, help='Gender')
    parser.add_argument('--visualize', required=False, action='store_true', help='Shows visualization of the fitting process.')
    parser.add_argument('--saveonly', required=False, action='store_true', help='Saves joints to files if results already exist.')
    parser.add_argument('--rotation', required=False, default=0, type=int, choices=[0, 90, 180, 270],
                        help='If camera is rotated set degrees of rotation here.')

    args = parser.parse_args()

    run_fitting(data_folder=args.data_folder,
                output_folder=args.output_folder,
                gender=args.gender,
                cam_rotation=args.rotation,
                do_viz=args.visualize,
                do_save_joints=args.saveonly)

    # write joints files after all fittings are done...
    if not args.saveonly:
        run_fitting(data_folder=args.data_folder,
                    output_folder=args.output_folder,
                    gender=args.gender,
                    cam_rotation=args.rotation,
                    do_viz=False,
                    do_save_joints=True)
