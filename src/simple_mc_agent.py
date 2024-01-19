"""
Class to generate patients given a set of parameters
"""
from typing import Dict
import numpy as np

# pylint: disable=too-many-function-args


def generate_patients(
    patient_profiles: Dict[str, np.array],
    num_patients: int,
) -> Dict[str, np.array]:
    """
    Generate patients given a set of parameters

    Parameters
    ----------
    patient_profiles: Dict[str, np.array]:
        Dictionary with the following keys:
            resource_probs : np.array
                Array of probabilities that a patient needs a resource
                shape = (M, K), M is the number of patient profiles, K is the types of resources
                Each values should be between 0 and 1 inclusive
            mean_times : np.array
                Array of mean times for each patient, this time represents the number of mins after
                which the patient's survival start dramatically decreasing
                shape = (M, 1), M is the number of patient profiles
            prob_patient_types : np.array
                Array of probabilities that a patient is of a certain type
                shape = (M, 1), M is the number of patient profiles
                Sum of all values should be 1
    num_patients : int
        Number of patients to generate

    Returns
    -------
    Dictionary containing the following keys:
        types: np.array
            Array of patient types
            shape = (num_patients, 1)
            Each value should be between 0 and M-1 inclusive
        resources: np.array
            Array of patient resources
            shape = (num_patients, K)
            Each value is either 0 or 1
        times: np.array
            Array of patient times
            shape = (num_patients, 1)
            Each value is a number of mins
    """
    resource_probs = patient_profiles["resource_probs"]
    mean_times = patient_profiles["mean_times"]
    prob_patient_types = patient_profiles["prob_patient_types"]

    types = np.random.choice(
        len(prob_patient_types), size=num_patients, p=prob_patient_types.reshape(-1)
    )

    mean_times = mean_times[types]
    sample_times = np.random.exponential(mean_times)

    rand_vals = np.random.uniform(0, 1, (num_patients, len(resource_probs[0])))
    resources = (rand_vals < resource_probs[types]).astype(int)

    return {
        "types": types.reshape(num_patients, 1),
        "resources": resources.reshape(num_patients, -1),
        "times": sample_times.reshape(num_patients, 1),
    }


def eval_prob_survivals(
    hospital_profiles: Dict[str, np.array],
    patients_sampled: Dict[str, np.array],
    assignments: np.array,
) -> float:
    """
    Evaluates the summation of the probability of survival for all patients given a particular assignment
    of patients to hospitals

    Parameters
    ----------
    hospital_profiles: Dict[str, np.array]
        Dictionary with the following keys:
            curr_resources : np.array
                Array with the current resources of each hospital
                shape = (N, K), N is the number of hospitals, K is the number of resources
            times_to_hospitals : np.array
                Array with the times to each hospital
                shape = (N, 1), N is the number of hospitals
            cost_no_resource: float
                Prob of survival is multiplied by this numbre for each resource not met
                Should be a value between 0 and 1
    patients_sampled : Dict[str, np.array]
        Dictionary defined according to the output of generate_patients
    assignments : np.array
        Array with the assignment of patients to hospitals
        shape = (M, 1), M is the number of patients
        Each value is a number between 0 and N-1 inclusive

    Returns
    -------
    float
        The summation of the probability of survival for all patients given a particular assignment
    """
    curr_resources = hospital_profiles["curr_resources"]
    times_to_hospitals = hospital_profiles["times_to_hospitals"]
    cost_no_resource = hospital_profiles["cost_no_resource"]

    n_patients, n_resources = patients_sampled["resources"].shape
    n_hospitals = curr_resources.shape[0]
    hosp_patient_resources = np.ones((n_hospitals, n_patients, n_resources))
    hosp_patient_resources *= patients_sampled["resources"].reshape(
        1, n_patients, n_resources
    )
    assignment_one_hot = np.zeros((n_patients, n_hospitals))
    assignment_one_hot[np.arange(n_patients), assignments.flatten()] = 1
    hosp_patient_resources *= assignment_one_hot.T.reshape(n_hospitals, n_patients, 1)
    cumulative_resources = np.cumsum(
        hosp_patient_resources, axis=1
    ) * assignment_one_hot.T.reshape(n_hospitals, n_patients, 1)
    resources_left_per_patient = (
        curr_resources.reshape(n_hospitals, 1, n_resources) - cumulative_resources
    )
    resources_left_per_patient *= hosp_patient_resources
    resources_left_per_patient[resources_left_per_patient >= 0] = 1
    resources_left_per_patient[resources_left_per_patient < 0] = cost_no_resource
    log_prob_survivals = np.log(resources_left_per_patient)
    log_resources_surv = log_prob_survivals.sum()

    arrival_times = times_to_hospitals[assignments.flatten()]
    log_times_surv = patients_sampled["times"] - arrival_times.reshape(n_patients, 1)
    log_times_surv[log_times_surv >= 0] = 0
    log_times_surv = log_times_surv.sum()

    return log_resources_surv + log_times_surv


def eval_state(
    patient_profiles: Dict[str, np.array],
    hospital_profiles: Dict[str, np.array],
    n_patients: int,
    n_patient_samples: int,
    n_assignment_samples: int,
) -> float:
    """
    Evaluates the state of the system given a set of parameters

    Parameters
    ----------
    patient_profiles : Dict[str, np.array]
        Dictionary containing the following keys:
            resource_probs: np.array
                Array of probabilities that a patient needs a resource
                shape = (M, K), M is the number of patient profiles, K is the types of resources
                Each values should be between 0 and 1 inclusive
            mean_times: np.array
                Array of mean times for each patient, this time represents the number of mins after
                which the patient's survival start dramatically decreasing
                shape = (M, 1), M is the number of patient profiles
            prob_patient_types: np.array
                Array of probabilities that a patient is of a certain type
                shape = (M, 1), M is the number of patient profiles
                Sum of all values should be 1
    hospital_profiles : Dict[str, np.array]
        Dictionary containing the following keys:
            curr_resources: np.array
                Array with the current resources of each hospital
                shape = (N, K), N is the number of hospitals, K is the number of resources
            times_to_hospitals: np.array
                Array with the times to each hospital
                shape = (N, 1), N is the number of hospitals
            cost_no_resource: float
                Prob of survival is multiplied by this numbre for each resource not met
                Should be a value between 0 and 1
    n_patients : int
        Number of patients left to assign
    n_patient_samples : int
        Number of patients to sample
    n_assignment_samples : int
        Number of assignments to sample for patient sample

    Returns
    -------
    float
        Prob of survival expected for the system
    """
    if n_patients == 0:
        return 0
    n_resources = hospital_profiles["curr_resources"].shape[1]

    eval_estimates = []
    for _ in range(n_patient_samples):
        patients_sampled = generate_patients(
            patient_profiles=patient_profiles,
            num_patients=n_patients,
        )
        estimates = []

        for _ in range(n_assignment_samples):
            assignments = np.random.randint(0, n_resources, size=(n_patients, 1))
            estimates.append(
                eval_prob_survivals(
                    hospital_profiles=hospital_profiles,
                    patients_sampled=patients_sampled,
                    assignments=assignments,
                )
            )
        eval_estimates.append(np.max(estimates))
    return np.mean(eval_estimates)


def find_best_assignment(
    patient_profiles: Dict[str, np.array],
    hospital_profiles: Dict[str, np.array],
    curr_patient_time: float,
    curr_patient_resources: np.array,
    n_patient_after_cur: int,
) -> int:
    """
    Find the best hospital to assign a patient to given the current state of the system

    Parameters
    ----------
    patient_profiles : Dict[str, np.array]
        As described in eval_state
    hospital_profiles : Dict[str, np.array]
        As described in eval_state
    curr_patient_time : float
        The curr time a patient has before their survival starts decreasing
    curr_patient_resources : np.array
        The curr resources a patient needs
    n_patient_after_cur : int
        The number of patients left to assign after the current patient

    Returns
    -------
    int
        The index of the hospital to assign the patient to
    """
    q_values = []

    for i in range(hospital_profiles["curr_resources"].shape[0]):
        new_resources = hospital_profiles["curr_resources"].copy()
        cur_patient_cost = new_resources[i] - curr_patient_resources.reshape(-1)
        cur_patient_cost = cur_patient_cost.astype(float)
        cur_patient_cost[cur_patient_cost > -1] = 1
        cur_patient_cost[cur_patient_cost < 0] = hospital_profiles["cost_no_resource"]
        cur_patient_cost = np.sum(np.log(cur_patient_cost))

        cur_patient_time_cost = (
            curr_patient_time - hospital_profiles["times_to_hospitals"][i]
        )
        if cur_patient_time_cost < 0:
            cur_patient_cost += cur_patient_time_cost[0]

        new_resources[i] -= curr_patient_resources.reshape(-1)
        new_resources[new_resources < 0] = 0
        rest_of_system = eval_state(
            patient_profiles=patient_profiles,
            hospital_profiles={
                "curr_resources": new_resources,
                "times_to_hospitals": hospital_profiles["times_to_hospitals"],
                "cost_no_resource": hospital_profiles["cost_no_resource"],
            },
            n_patients=n_patient_after_cur,
            n_patient_samples=10,
            n_assignment_samples=100,
        )

        q_values.append(cur_patient_cost + rest_of_system)
    return np.argmax(q_values)
