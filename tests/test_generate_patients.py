"""
Generate patient function unittests
"""
import unittest

import numpy as np
from src.simple_mc_agent import generate_patients


class TestGeneratePatients(unittest.TestCase):
    """
    Test generate_patients function
    """

    def test_generate_patients(self):
        """
        Simple test
        2 patient types
        3 resource types
        """
        np.random.seed(0)
        resource_probs = np.array([[0.5, 0.5, 0.5], [0.99, 0.01, 0.99]])
        mean_times = np.array([[10], [100]])
        prob_patient_types = np.array([[0.8], [0.2]])
        patient_profiles = {
            "resource_probs": resource_probs,
            "mean_times": mean_times,
            "prob_patient_types": prob_patient_types,
        }
        patients_sampled = generate_patients(patient_profiles, 10000)

        # check prob patient types
        estimated_0 = (patients_sampled["types"] == 0).sum() / 10000
        estimated_1 = (patients_sampled["types"] == 1).sum() / 10000
        print(np.abs(estimated_0 - prob_patient_types[0][0]))
        print(np.abs(estimated_1 - prob_patient_types[1][0]))
        self.assertTrue(np.abs(estimated_0 - prob_patient_types[0][0]) < 0.05)
        self.assertTrue(np.abs(estimated_1 - prob_patient_types[1][0]) < 0.05)

        # check mean times
        mean_0 = patients_sampled["times"][patients_sampled["types"] == 0].mean()
        mean_1 = patients_sampled["times"][patients_sampled["types"] == 1].mean()
        print(np.abs(mean_0 - mean_times[0][0]))
        print(np.abs(mean_1 - mean_times[1][0]))
        self.assertTrue(np.abs(mean_0 - mean_times[0][0]) < 1)
        self.assertTrue(np.abs(mean_1 - mean_times[1][0]) < 2)

        # check resource probs
        resources_0 = patients_sampled["resources"][
            patients_sampled["types"].reshape(-1) == 0
        ].mean(axis=0)
        resources_1 = patients_sampled["resources"][
            patients_sampled["types"].reshape(-1) == 1
        ].mean(axis=0)
        print(np.abs(resources_0 - resource_probs[0]))
        print(np.abs(resources_1 - resource_probs[1]))
        self.assertTrue(np.all(np.abs(resources_0 - resource_probs[0]) < 0.05))
        self.assertTrue(np.all(np.abs(resources_1 - resource_probs[1]) < 0.05))


if __name__ == "__main__":
    unittest.main()
