"""
Generate eval_prob_survivals function unittests
"""
import unittest

import numpy as np
from src.simple_mc_agent import eval_prob_survivals


class TestEvalProbSurvivals(unittest.TestCase):
    """
    Test eval_prob_survivals function
    """

    def test_eval_prob_survivals(self):
        """
        Simple test
        Patients sampled are 3 patients, 2 resource types
        4 hospitals with different mean time and resources at the start
        """
        np.random.seed(0)
        patient_sampled_resources = np.array([[1, 0], [0, 1], [1, 1]])
        patient_times = np.array([[10], [20], [15]])
        patients_sampled = {
            "types": None,
            "times": patient_times,
            "resources": patient_sampled_resources,
        }

        hospital_times = np.array([[8], [12], [17], [22]])
        hospital_resources = np.array([[2, 3], [3, 1], [1, 3], [3, 2]])
        cost_no_resource = 0.5
        hospital_profiles = {
            "curr_resources": hospital_resources,
            "times_to_hospitals": hospital_times,
            "cost_no_resource": cost_no_resource,
        }
        assignment_1 = np.array([0, 2, 1])
        expected_value_1 = 0  # manually computed
        pred_value_1 = eval_prob_survivals(
            hospital_profiles, patients_sampled, assignment_1
        )
        print(pred_value_1, expected_value_1)
        self.assertEqual(pred_value_1, expected_value_1)

        assignment_2 = np.array([1, 1, 1])
        expected_value_2 = np.log(0.5) - 2  # manually computed
        pred_value_2 = eval_prob_survivals(
            hospital_profiles, patients_sampled, assignment_2
        )
        print(pred_value_2, expected_value_2)
        self.assertEqual(pred_value_2, expected_value_2)


if __name__ == "__main__":
    unittest.main()
