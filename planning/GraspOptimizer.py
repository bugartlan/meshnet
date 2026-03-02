from abc import ABC

import numpy as np


class GraspOptimizer(ABC):
    """
    Abstract class for optimizing grasps on a given mesh.
    """

    def __init__(self, gripper, config):
        self.gripper = gripper
        self.config = config

    def optimize(self, mesh):
        raise NotImplementedError


class HeuristicGraspOptimizer(GraspOptimizer):
    """
    Class for optimizing grasps on a given mesh using heuristics.
    """

    def __init__(self, gripper, config):
        """
        Initializes a grasp optimizer.

        Parameters
        ----------
        mesh : Trimesh.Trimesh
            Mesh of the object. Assume the adhesive layer is z = 0.

        gripper : Gripper
            Mesh of the gripper used for collision checks.

        config : dict
            Configurations for the optimizer
        """
        super().__init__(gripper, config)
        self.friction_coeff = config["friction_coeff"]

        # Weights
        self.w_prob_force_closure = (
            1
            if "w_prob_force_closure" not in config
            else config["w_prob_force_closure"]
        )
        self.w_d_max = 1 if "w_d_max" not in config else config["w_d_max"]
        self.w_S_min = 1 if "w_S_min" not in config else config["w_S_min"]

        if "control_noise_stds" in config:
            # Number of noise samples
            self.n_noise = 10
            self.control_noise_sampler = GaussianControlNoiseSampler(
                stds=config["control_noise_stds"]
            )
        else:
            # No noise
            self.n_noise = 1
            self.control_noise_sampler = GaussianControlNoiseSampler(stds=np.zeros(6))

        self.n_segments = 10
        self.bending_direction_cache = None

    def _second_moment_of_area(self, z):
        """
        Compute the second moment of area at a given z-plane.

        https://en.wikipedia.org/wiki/Second_moment_of_area
        """
        planar, _ = self.mesh.section(
            plane_origin=[0, 0, z], plane_normal=[0, 0, 1]
        ).to_2D()

        weighted_centroids = 0
        total_area = 0
        Ix, Iy, Ixy = 0, 0, 0
        for poly in planar.polygons_full:
            coords = np.array(poly.exterior.coords)
            x = coords[:, 0]
            y = coords[:, 1]
            a = x[:-1] * y[1:] - x[1:] * y[:-1]
            Ix += np.sum((y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) * a) / 12
            Iy += np.sum((x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) * a) / 12
            Ixy += (
                np.sum(
                    (
                        2 * x[:-1] * y[:-1]
                        + x[:-1] * y[1:]
                        + x[1:] * y[:-1]
                        + 2 * x[1:] * y[1:]
                    )
                    * a
                )
                / 24
            )
            area = np.sum(a) / 2
            total_area += area
            weighted_centroids += area * np.array(poly.centroid.coords[0])
        centroid = weighted_centroids / total_area
        Ix -= total_area * centroid[1] ** 2
        Iy -= total_area * centroid[0] ** 2
        Ixy -= total_area * centroid[0] * centroid[1]
        return Ix, Iy, Ixy, total_area, centroid

    def _leverarm(self, interface, grasp):
        """
        Compute the longest distance from the grasp to the edges of the interface.

        Parameters
        ----------
        interface : Trimesh.Trimesh
            Interface mesh of the object

        grasp : AntipodalGrasp
            Grasp to be evaluated

        Returns
        -------
        * float
            Score for the bending moment heuristics. The longest distance from the grasp to any edges.
        """
        d_max = 0

        for poly in interface.polygons_full:
            coords = np.array(poly.exterior.coords)
            for i in range(len(coords) - 1):
                # p1, p2 are two endpoints of an edge
                p1 = np.concatenate([coords[i], [0]])
                p2 = np.concatenate([coords[i + 1], [0]])
                t = p2 - p1  # tangent vector
                # Project the the vector from grasp.mid to p1 to the edge
                proj_g = np.dot(grasp.mid - p1, t) / np.dot(t, t) * t
                d = np.sqrt(
                    np.dot(grasp.mid - p1, grasp.mid - p1) - np.dot(proj_g, proj_g)
                )
                if d > d_max:
                    d_max = d
                    self.bending_direction_cache = np.cross(t, grasp.mid - p1)
                    self.bending_direction_cache /= np.linalg.norm(
                        self.bending_direction_cache
                    )
                    if np.dot(self.bending_direction_cache, [0, 0, 1]) < 0:
                        self.bending_direction_cache = -self.bending_direction_cache

        return d_max

    def _xarea(self, slices, grasp):
        """
        Compute the cross-sectional area below the grasp.

        Parameters
        ----------
        slices : list of tuples
            Each tuple contains (z, section) where section is (Ix, Iy, Ixy, area, centroid)

        grasp : AntipodalGrasp
            Grasp to be evaluated

        Returns
        -------
        * float
            Minimum cross-sectional area below the grasp
        """
        return min((area for z, area in slices if z <= grasp.mid[-1]), default=None)

    def _evaluate(self, grasp, interface, slices, debug=False):
        """
        Evaluate the grasp.

        Parameters
        ----------
        grasp : AntipodalGrasp
            Grasp to be evaluated

        debug : bool, optional
            If True, prints debug information (default is False)

        Returns
        -------
        * float
            Score for the grasp
        """
        return self._weighted_score(
            grasp._validate(),
            self._leverarm(interface, grasp),
            self._xarea(slices, grasp),
            debug=debug,
        )
        total_score = 0
        n_fc = 0
        for i in range(self.n_noise):
            control_noise_mat = self.control_noise_sampler.sample()
            g_noisy = grasp.transform(
                control_noise_mat, sampler.mesh, self.friction_coeff
            )
            valid_grasp = False
            if (
                g_noisy is not None
                and sampler.sample_approach_direction(g_noisy) is not None
            ):
                d_max, S_min = self._bending_moment_score(sampler.mesh, g_noisy)
                total_score += self._weighted_score(g_noisy._validate(), d_max, S_min)
                n_fc += g_noisy._validate()
                valid_grasp = True
            if debug:
                if valid_grasp:
                    print(
                        f"Iteration #{i + 1}: fc = {g_noisy._validate()}, d_max = {d_max}, S_min = {S_min}, score = {total_score / (i + 1)}"
                    )
                    # visualize(self.mesh, g_noisy, self.gripper)
                else:
                    print("N/A")
        prob_force_closure = n_fc / self.n_noise
        if debug:
            print(
                f"Final score: {total_score / self.n_noise}, prob_force_closure = {prob_force_closure}"
            )
        return total_score / self.n_noise

    def _weighted_score(self, prob_force_closure, d_max, S_min, debug=False):
        """
        Compute the weighted score.

        Parameters
        ----------
        prob_force_clousre : float
            Probability of force closure

        d_max : float
            Longest distance from grasp to boundaries

        S_min : float
            Smallest cross sectional modulus below the grasp

        debug : bool, optional
            If True, prints debug information (default is False)

        Returns
        -------
        * float
            Weighted score
        """
        if debug:
            print(
                f"prob_force_closure: {prob_force_closure}, d_max: {d_max}, S_min: {S_min}"
            )
        return (
            prob_force_closure * self.w_prob_force_closure
            + d_max * self.w_d_max
            + S_min * self.w_S_min
        )

    def _area(self, mesh, z):
        """
        Compute the cross-sectional area at a given z-plane.

        Parameters
        ----------
        mesh : Trimesh.Trimesh
            Mesh of the object

        z : float
            z-coordinate of the plane

        Returns
        -------
        * float
            Cross-sectional area at the given z-plane
        """
        planar, _ = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1]).to_2D()
        return planar.area

    def evaluate(self, mesh, grasp):
        """
        Evaluate a given grasp on the mesh.

        Parameters
        ----------
        mesh : Trimesh.Trimesh
            Mesh of the object. Assume the adhesive layer is z = 0.

        grasp : AntipodalGrasp
            Grasp to be evaluated

        Returns
        -------
        * float
            Score for the grasp
        """
        # Extract the interface polygon
        interface, _ = mesh.section(
            plane_origin=[0, 0, 0.001], plane_normal=[0, 0, 1]
        ).to_2D()
        if interface is None:
            raise ValueError("The given mesh doesn't have an interface at z=0.")

        # Compute the second moment of areas along the z-axis
        slices = [
            (z, self._area(mesh, z))
            for z in np.linspace(0, mesh.bounds[1][2], self.n_segments, endpoint=False)
        ]

        return self._evaluate(grasp, interface, slices, debug=False)

    def optimize(self, mesh, n_samples=100, debug=False):
        """
        Search for an optimal grasp.

        Parameters
        ----------
        mesh : Trimesh.Trimesh
            Mesh of the object. Assume the adhesive layer is z = 0.

        n_samples : int
            Number of samples

        debug : bool, optional
            If True, prints debug information (default is False)

        Returns
        -------
        * AntipodalGrasp
            Optimal grasp
        """

        # Extract the interface polygon
        interface, _ = mesh.section(
            plane_origin=[0, 0, 0.001], plane_normal=[0, 0, 1]
        ).to_2D()
        if interface is None:
            raise ValueError("The given mesh doesn't have an interface at z=0.")

        # Compute the second moment of areas along the z-axis
        slices = [
            (z, self._area(mesh, z))
            for z in np.linspace(0, mesh.bounds[1][2], self.n_segments, endpoint=False)
        ]

        if debug:
            print(f"Sampling {n_samples} candidate grasps...")
        sampler = AntipodalGraspSampler(self.gripper, mesh, self.config)
        candidates = sampler.sample(n_samples, max_trials=n_samples * 10)
        optimal_grasp = None
        best_score = -np.inf
        if debug:
            print("Evaluating candidate grasps...")
        for g in candidates:
            score = self._evaluate(g, interface, slices, debug=debug)
            if best_score < score:
                optimal_grasp = g
                best_score = score

        return optimal_grasp, best_score


class RandomGraspOptimizer(GraspOptimizer):
    """
    Class for optimizing grasps on a given mesh using random sampling.
    """

    def __init__(self, gripper, config):
        """
        Initializes a random grasp optimizer.

        Parameters
        ----------
        mesh : Trimesh.Trimesh
            Mesh of the object. Assume the adhesive layer is z = 0.

        gripper : Gripper
            Mesh of the gripper used for collision checks.

        config : dict
            Configurations for the optimizer
        """
        super().__init__(gripper, config)

    def optimize(self, mesh, n_samples=100, debug=False):
        """
        Search for an optimal grasp.

        Parameters
        ----------
        n_samples : int
            Number of samples

        Returns
        -------
        * AntipodalGrasp
            Optimal grasp
        """
        if debug:
            print(f"Sampling {n_samples} candidate grasps...")
        sampler = AntipodalGraspSampler(self.gripper, mesh, self.config)
        candidates = sampler.sample(1)
        if len(candidates) == 0:
            return None

        return np.random.choice(candidates), 0.0
