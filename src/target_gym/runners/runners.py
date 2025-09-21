from tqdm import tqdm

from target_gym.runners.bicycle_runner import run_all_modes as run_bike
from target_gym.runners.car_runner import run_all_modes as run_car
from target_gym.runners.cstr_runner import run_all_modes as run_cstr
from target_gym.runners.plane_runner import run_all_modes as run_plane

if __name__ == "__main__":
    environments = [run_plane, run_bike, run_car, run_cstr]
    for run_env in tqdm(environments):
        run_env()
