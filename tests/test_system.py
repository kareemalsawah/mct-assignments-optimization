"""
Generate find_best_assignment function unittests
"""
import unittest
import copy
import numpy as np
from src.simple_mc_agent import find_best_assignment, generate_patients


class SimEnv:
    def __init__(self, patient_profiles, hospital_profiles, n_patients):
        self.patient_profiles = patient_profiles
        self.hospital_profiles = hospital_profiles
        self.n_patients = n_patients
        self.actual_patients_sample = None
        self.cur_patient_time = None
        self.cur_patient_resources = None

    def reset(self):
        self.actual_patients_sample = generate_patients(
            self.patient_profiles, self.n_patients
        )
        patient_time, patient_resources = self.get_patient()
        self.cur_patient_time, self.cur_patient_resources = (
            patient_time,
            patient_resources,
        )
        return patient_time, patient_resources

    def get_patient(self):
        if self.actual_patients_sample["resources"].shape[0] == 0:
            self.actual_patients_sample = None
            self.cur_patient_time = None
            self.cur_patient_resources = None
            return None, None
        patient_resources = self.actual_patients_sample["resources"][0].reshape(-1)
        patient_time = self.actual_patients_sample["times"][0]

        # Remove patient from current
        self.actual_patients_sample["resources"] = self.actual_patients_sample[
            "resources"
        ][1:]
        self.actual_patients_sample["times"] = self.actual_patients_sample["times"][1:]

        return patient_time, patient_resources

    def step(self, action: int):
        if not self.actual_patients_sample:
            raise ValueError(
                "SimEnv: Environment ended or not initialized. Call .reset() before calling .step()"
            )

        # compute reward
        hospital_resources = self.hospital_profiles["curr_resources"][action]
        diff = hospital_resources - self.cur_patient_resources
        n_resources_not_met = diff[diff < 0].sum()
        log_prob_surv = n_resources_not_met * np.log(
            self.hospital_profiles["cost_no_resource"]
        )
        log_time_cost = 0
        diff_time = (
            self.cur_patient_time - self.hospital_profiles["times_to_hospitals"][action]
        )
        if diff_time < 0:
            log_time_cost = diff_time

        reward = log_prob_surv + log_time_cost

        # remove resources from hospital
        self.hospital_profiles["curr_resources"][action] -= self.cur_patient_resources
        self.hospital_profiles["curr_resources"][action][
            self.hospital_profiles["curr_resources"][action] < 0
        ] = 0

        is_done = False
        patient_time, patient_resources = self.get_patient()
        if patient_time is None:
            is_done = True
        self.cur_patient_time, self.cur_patient_resources = (
            patient_time,
            patient_resources,
        )
        return (patient_time, patient_resources), reward, is_done


class TestFindBestAssignment(unittest.TestCase):
    """
    Test find_best_assignment function
    """

    def test_find_best_assignment(self):
        """
        Simple test
        Patients sampled are 3 patients, 2 resource types
        4 hospitals with different mean time and resources at the start
        """
        np.random.seed(0)
        n_patients = 20
        resource_probs = np.array([[0.3, 0.5, 0.7], [0.99, 0.01, 0.99]])
        mean_times = np.array([[10], [20]]) * 2
        prob_patient_types = np.array([[0.8], [0.2]])
        patient_profiles = {
            "resource_probs": resource_probs,
            "mean_times": mean_times,
            "prob_patient_types": prob_patient_types,
        }

        n_hospitals = 4
        hospital_times = np.array([[8], [12], [17], [22]]) * 0.2
        hospital_resources = np.array([[2, 3, 4], [3, 1, 2], [1, 3, 6], [3, 2, 3]]) * 4
        cost_no_resource = 0.5
        hospital_profiles = {
            "curr_resources": hospital_resources,
            "times_to_hospitals": hospital_times,
            "cost_no_resource": cost_no_resource,
        }

        # Run random assignments
        probs = []
        for _ in range(10):
            env = SimEnv(patient_profiles, copy.deepcopy(hospital_profiles), n_patients)
            obs = env.reset()
            done = False
            total_probs = 0
            while not done:
                action = np.random.randint(0, n_hospitals)
                obs, reward, done = env.step(action)
                total_probs += reward.item()

            probs.append(total_probs)
        random_agent_mean = np.mean(probs)
        print("Random agent: ", random_agent_mean)

        # Run optimal agent
        probs = []
        for i in range(10):
            env = SimEnv(patient_profiles, copy.deepcopy(hospital_profiles), n_patients)
            print(i)
            obs = env.reset()
            done = False
            total_probs = 0
            while not done:
                n_next_patients = env.actual_patients_sample["resources"].shape[0]
                action = find_best_assignment(
                    patient_profiles,
                    env.hospital_profiles,
                    obs[0],
                    obs[1],
                    n_next_patients,
                )
                obs, reward, done = env.step(action)
                total_probs += reward.item()

            probs.append(total_probs)
        optimal_agent_mean = np.mean(probs)
        print("Optimal agent: ", optimal_agent_mean)
        self.assertTrue(optimal_agent_mean > random_agent_mean)


if __name__ == "__main__":
    unittest.main()
