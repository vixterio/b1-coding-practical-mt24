from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
               # Open the CSV file and inspect the first non-empty line to decide if there is a header.
        import csv  # local import so top-of-file imports remain unchanged
        with open(file_name, 'r', encoding='utf-8') as fh:
            # Read lines until we find a non-empty one (skip leading blank lines).
            first_line = ''
            for line in fh:
                stripped = line.strip()
                if stripped:
                    first_line = stripped
                    break

        # If no non-empty line was found, the file is empty -> raise.
        if not first_line:
            raise ValueError(f"Empty CSV file: {file_name}")

        # Use csv.reader on the first line to split it into fields respecting quoting.
        first_row = next(csv.reader([first_line]))
        # Try to convert each token to float; if any fails, treat the row as a header.
        has_header = False
        for tok in first_row:
            try:
                float(tok)
            except Exception:
                has_header = True
                break

        # If there is no header: treat the file as purely numeric with columns in order:
        # reference, cave_height, cave_depth (first three columns).
        if not has_header:
            # Load all numeric data; loadtxt will error if non-numeric rows exist.
            arr = np.loadtxt(file_name, delimiter=',')
            # Ensure shape is 2-D even for single-row files.
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Require at least three columns.
            if arr.shape[1] < 3:
                raise ValueError("CSV must contain at least three numeric columns: reference, cave_height, cave_depth")
            # Map columns by position.
            reference = arr[:, 0].astype(float)
            cave_height = arr[:, 1].astype(float)
            cave_depth = arr[:, 2].astype(float)
            # Return a Mission instance built from the numeric columns.
            return cls(reference, cave_height, cave_depth)

        # If we reach here, the first row looked like a header -> use DictReader to load columns by name.
        with open(file_name, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            # Collect rows into lists keyed by normalized column names.
            cols = {}
            # Normalize header names to a canonical form for easier matching.
            fieldnames = reader.fieldnames or []
            norm_map = {}
            for name in fieldnames:
                # create a normalized key: lowercased, spaces and hyphens -> underscores, stripped
                key = name.lower().strip().replace(' ', '_').replace('-', '_')
                norm_map[key] = name  # map normalized -> original header

            # Prepare candidate logical names we accept for each mission array.
            ref_candidates = ['reference', 'ref']
            height_candidates = ['cave_height', 'height', 'ceiling', 'cave_ceiling']
            depth_candidates = ['cave_depth', 'depth', 'floor', 'cave_floor']

            # Helper to select the first matching original header name from candidates.
            def pick_field(candidates):
                for cand in candidates:
                    if cand in norm_map:
                        return norm_map[cand]
                return None

            # Pick actual header names present in the file for each logical column.
            ref_field = pick_field(ref_candidates)
            height_field = pick_field(height_candidates)
            depth_field = pick_field(depth_candidates)

            # If any required column is missing, raise a clear error listing available fields.
            if ref_field is None or height_field is None or depth_field is None:
                raise ValueError(f"CSV missing required columns. Available: {fieldnames}")

            # Initialize lists to accumulate column values.
            ref_list = []
            height_list = []
            depth_list = []

            # Iterate rows and append converted floats into the lists.
            for row in reader:
                try:
                    ref_list.append(float(row[ref_field]))
                    height_list.append(float(row[height_field]))
                    depth_list.append(float(row[depth_field]))
                except Exception as exc:
                    # If conversion fails for any row, raise an informative error.
                    raise ValueError(f"Non-numeric value encountered in CSV: {exc}")

            # Convert lists to numpy arrays of dtype float and return a Mission instance.
            reference = np.asarray(ref_list, dtype=float)
            cave_height = np.asarray(height_list, dtype=float)
            cave_depth = np.asarray(depth_list, dtype=float)
            return cls(reference, cave_height, cave_depth)
        pass


class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()
        # if controller provides reset(), call it so internal state starts fresh
        # (stateful controllers like PDController implemented above provide reset())
        if hasattr(self.controller, "reset"):
            self.controller.reset()


        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            reference_t = mission.reference[t]
            dt = getattr(self.plant, "dt", 1.0)  # sampling time from plant, default to 1.0
            if hasattr(self.controller, "step"):
                # stateful controller path (recommended)
                action_t = self.controller.step(reference_t, observation_t, dt)
            else:
                # stateless/callable controller path: call with (reference, measurement, dt)
                action_t = self.controller(reference_t, observation_t, dt)

            # store action and apply transition with disturbance
            actions[t] = action_t
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
